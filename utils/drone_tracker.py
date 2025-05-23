\
import numpy as np
from ultralytics import YOLO

# Attempt to import ByteTrack
try:
    from yolox.tracker.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    # Define a dummy BYTETracker class if the import fails
    class BYTETrackerDummy: 
        def __init__(self, *args, **kwargs):
            print("Warning: ByteTrack not found or import failed. Tracking will be basic.")
        def update(self, *args, **kwargs):
            print("Warning: ByteTrack not available, returning empty tracks.")
            return [] 

class DroneTracker:
    def __init__(self, yolo_model_path, 
                 # ByteTrack does not require a separate ReID model path in the same way DeepSORT does.
                 # Common ByteTrack parameters:
                 track_thresh=0.5, # Tracking confidence threshold
                 track_buffer=30,  # Buffer to keep lost tracks (frames)
                 match_thresh=0.8, # Matching threshold for IOU
                 frame_rate=30,
                 device='cuda'):
        """
        Initializes the DroneTracker with YOLO model and ByteTrack.

        Args:
            yolo_model_path (str): Path to the YOLO model file.
            track_thresh (float): Tracking confidence threshold.
            track_buffer (int): Number of frames to buffer tracks before deletion.
            match_thresh (float): IOU threshold for matching tracks with detections.
            frame_rate (int): Frame rate of the video.
            device (str): 'cuda' or 'cpu'. (YOLO uses this, ByteTrack is CPU based but works on detections from GPU)
        """
        print(f"Loading YOLO model from {yolo_model_path}...")
        self.yolo_model = YOLO(yolo_model_path)
        if device == 'cuda':
            self.yolo_model.to('cuda')
        print("YOLO model loaded.")

        self.tracker = None
        if BYTETRACK_AVAILABLE:
            print("Initializing ByteTrack tracker...")
            try:
                self.tracker = BYTETracker( 
                    track_thresh=track_thresh,
                    track_buffer=track_buffer,
                    match_thresh=match_thresh,
                    # The BYTETracker class from yolox might take args directly or an args object
                    # Depending on the exact version of bytetracker, you might need to pass an argparse.Namespace
                    # e.g. args = argparse.Namespace(track_thresh=0.5, ...); self.tracker = BYTETracker(args)
                    frame_rate=frame_rate
                )
                print("ByteTrack tracker initialized successfully.")
            except Exception as e:
                print(f"Error initializing ByteTrack: {e}")
                print("ByteTrack tracking will be disabled.")
                self.tracker = None
        else:
            print("ByteTrack library not found. Tracking will rely on detections only (no persistent IDs).")
            # If ByteTrack is not available, we might want to ensure self.tracker is explicitly None
            # or assign the dummy tracker if we had one that mimicked the interface for basic operation.
            # For now, the logic correctly handles self.tracker being None.

    def update(self, image):
        """
        Detects drones in the image and updates their tracks using ByteTrack.

        Args:
            image (ndarray): Input image (BGR format expected by OpenCV and many models).

        Returns:
            tuple: (tracked_objects, raw_detections_xywh)
                tracked_objects (ndarray): Array of [x_center, y_center, w, h, track_id].
                                           If tracking is disabled or fails, track_id might be a simple index.
                raw_detections_xywh (ndarray): Array of [x_center, y_center, w, h] from YOLO.
        """
        # 1. YOLO Detection
        yolo_results = self.yolo_model(image, verbose=False)
        
        raw_detections_xywh_list = []
        # ByteTrack expects detections in [x1, y1, x2, y2, score] format
        detections_for_bytetrack = [] 

        if yolo_results and yolo_results[0].boxes:
            raw_detections_xywh_list = yolo_results[0].boxes.xywh.cpu().numpy()
            
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                # cls_id = int(box.cls[0].item()) # If you need class ID
                detections_for_bytetrack.append([x1, y1, x2, y2, conf])
        
        raw_detections_xywh = np.array(raw_detections_xywh_list) if raw_detections_xywh_list else np.empty((0, 4))
        detections_for_bytetrack_np = np.array(detections_for_bytetrack) if detections_for_bytetrack else np.empty((0, 5))

        # 2. ByteTrack Tracking
        if self.tracker and BYTETRACK_AVAILABLE and detections_for_bytetrack_np.shape[0] > 0:
            try:
                # ByteTrack's update method expects detections and the image shape (or other info like image size)
                # The exact signature might vary slightly based on the specific bytetracker implementation.
                # This is a common way to call it:
                online_targets = self.tracker.update(detections_for_bytetrack_np, image.shape[:2])
                # online_targets is a list of STTrack objects. Each STTrack object has a tlwh attribute and track_id.
                # tlwh is [top_left_x, top_left_y, width, height]
            except Exception as e:
                print(f"Error during ByteTrack update: {e}")
                online_targets = [] # Fallback to empty if error

            tracked_objects_list = []
            if online_targets:
                for track in online_targets:
                    tlwh = track.tlwh
                    track_id = track.track_id
                    # Convert tlwh to center_x, center_y, w, h
                    center_x = tlwh[0] + tlwh[2] / 2
                    center_y = tlwh[1] + tlwh[3] / 2
                    width = tlwh[2]
                    height = tlwh[3]
                    tracked_objects_list.append([center_x, center_y, width, height, int(track_id)])
                
                tracked_objects_final = np.array(tracked_objects_list)
                return tracked_objects_final, raw_detections_xywh
            else:
                # No tracks output from ByteTrack
                return np.empty((0, 5)), raw_detections_xywh
        else:
            # Tracking disabled or no detections to track.
            # Return raw detections with a dummy track_id (index) for consistent format.
            tracked_objects_list = []
            for i, det_xywh in enumerate(raw_detections_xywh):
                tracked_objects_list.append(np.append(det_xywh, i + 1)) # Use index as dummy track_id
            
            tracked_objects_final = np.array(tracked_objects_list) if tracked_objects_list else np.empty((0,5))
            return tracked_objects_final, raw_detections_xywh
