"""
SteroDrone 主程序 - 使用双目视觉检测并定位无人机

该程序使用ZED2i立体相机采集图像，通过自定义双目立体匹配和三角测量计算深度，
结合目标检测模型来检测、定位无人机，并将其坐标从相机坐标系转换到全球地理坐标系。

主要功能:
1. 初始化ZED2i相机（仅用于图像采集，不使用内置深度功能）
2. 读取相机标定参数
3. 双目图像采集
4. 使用YOLO模型进行无人机检测
5. 立体匹配与三维重建（使用自定义算法）
6. 坐标系转换（相机坐标系 -> 全球地理坐标系）
7. 结果可视化显示
"""

import cv2
import numpy as np
import time
import yaml
import pyzed.sl as sl  # ZED SDK Python API
from ultralytics import YOLO
from pyproj import Proj, Transformer, CRS
from scipy.spatial.transform import Rotation as R
from utils.geo_transform import CoordinateTransformer
from utils.stereo_matcher import StereoMatcher
from utils.drone_tracker import DroneTracker # Added import

def load_calibration_data(zed_camera=None, calibration_file=None):
    """
    从ZED相机获取标定参数，或从自定义标定文件加载
    
    参数:
        zed_camera (sl.Camera, optional): 初始化后的ZED相机对象
        calibration_file (str, optional): 备用标定文件路径，仅在无法从相机获取参数时使用
        
    返回:
        tuple: 包含以下标定参数的元组，若加载失败则返回None
            - cam_matrix_left (ndarray): 左相机内参矩阵
            - dist_coeffs_left (ndarray): 左相机畸变系数
            - cam_matrix_right (ndarray): 右相机内参矩阵
            - dist_coeffs_right (ndarray): 右相机畸变系数
            - R (ndarray): 右相机相对于左相机的旋转矩阵
            - T (ndarray): 右相机相对于左相机的平移向量
            - R1 (ndarray): 左相机校正旋转矩阵
            - R2 (ndarray): 右相机校正旋转矩阵
            - P1 (ndarray): 左相机投影矩阵
            - P2 (ndarray): 右相机投影矩阵
            - Q (ndarray): 视差-深度映射矩阵
            - image_size (tuple): 图像尺寸 (width, height)
            - roi_left (tuple): 左图像有效区域
            - roi_right (tuple): 右图像有效区域
    """
    if zed_camera is not None:
        try:
            # 从ZED相机获取标定参数
            calibration_params = zed_camera.get_camera_information().camera_configuration.calibration_parameters
            
            # 左相机参数
            cam_matrix_left = np.array([
                [calibration_params.left_cam.fx, 0, calibration_params.left_cam.cx],
                [0, calibration_params.left_cam.fy, calibration_params.left_cam.cy],
                [0, 0, 1]
            ])
            
            # ZED相机镜头畸变系数（通常很小，但为准确起见我们获取它们）
            dist_coeffs_left = np.array([
                calibration_params.left_cam.disto[0],
                calibration_params.left_cam.disto[1],
                calibration_params.left_cam.disto[2],
                calibration_params.left_cam.disto[3],
                calibration_params.left_cam.disto[5]
            ])
            
            # 右相机参数
            cam_matrix_right = np.array([
                [calibration_params.right_cam.fx, 0, calibration_params.right_cam.cx],
                [0, calibration_params.right_cam.fy, calibration_params.right_cam.cy],
                [0, 0, 1]
            ])
            
            dist_coeffs_right = np.array([
                calibration_params.right_cam.disto[0],
                calibration_params.right_cam.disto[1],
                calibration_params.right_cam.disto[2],
                calibration_params.right_cam.disto[3],
                calibration_params.right_cam.disto[5]
            ])
            
            # 相机之间的转换
            R = np.array(calibration_params.R)
            T = np.array(calibration_params.T)
            
            # 获取图像尺寸
            camera_resolution = zed_camera.get_camera_information().camera_configuration.resolution
            image_size = (camera_resolution.width, camera_resolution.height)
            
            # 计算校正参数
            R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
                cam_matrix_left, dist_coeffs_left,
                cam_matrix_right, dist_coeffs_right,
                image_size, R, T, alpha=0)

            print("从ZED相机加载标定参数成功。")
            return (cam_matrix_left, dist_coeffs_left, cam_matrix_right, dist_coeffs_right, 
                    R, T, R1, R2, P1, P2, Q, image_size, roi_left, roi_right)
                    
        except Exception as e:
            print(f"从ZED相机获取标定参数失败: {e}")
            if not calibration_file:
                return None
    
    # 如果没有相机对象或从相机获取参数失败，尝试从文件加载
    if calibration_file:
        try:
            with open(calibration_file, 'r') as f:
                calib_data = yaml.safe_load(f)
            
            cam_matrix_left = np.array(calib_data['camera_matrix_left'])
            dist_coeffs_left = np.array(calib_data['dist_coeffs_left'])
            cam_matrix_right = np.array(calib_data['camera_matrix_right'])
            dist_coeffs_right = np.array(calib_data['dist_coeffs_right'])
            R = np.array(calib_data['R'])
            T = np.array(calib_data['T'])
            image_size = tuple(calib_data['image_size'])
            
            # 计算校正参数
            R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
                cam_matrix_left, dist_coeffs_left,
                cam_matrix_right, dist_coeffs_right,
                image_size, R, T, alpha=0)
            
            # 如果标定文件包含ROI信息则使用
            if 'roi_left' in calib_data and 'roi_right' in calib_data:
                roi_left = tuple(calib_data['roi_left'])
                roi_right = tuple(calib_data['roi_right'])

            print("从标定文件加载参数成功。")
            return (cam_matrix_left, dist_coeffs_left, cam_matrix_right, dist_coeffs_right, 
                    R, T, R1, R2, P1, P2, Q, image_size, roi_left, roi_right)
                    
        except FileNotFoundError:
            print(f"错误：标定文件 {calibration_file} 未找到。")
            return None
        except Exception as e:
            print(f"加载标定文件时出错: {e}")
            return None
    
    return None

def init_zed_camera(resolution=sl.RESOLUTION.HD720, fps=30):
    """
    初始化ZED2i相机（不启用深度功能）
    
    参数:
        resolution (sl.RESOLUTION): 相机分辨率，默认HD720 (1280x720)
        fps (int): 帧率，默认30fps
        
    返回:
        sl.Camera: 初始化后的相机对象，如果初始化失败则返回None
    """
    # 创建相机对象
    zed = sl.Camera()
    
    # 设置初始化参数
    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.camera_fps = fps
    
    # 禁用深度计算，只使用相机进行图像采集
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # 打开相机
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"无法打开ZED相机: {err}")
        return None
        
    # 等待相机准备就绪
    print("正在预热ZED相机...")
    time.sleep(2.0)
    
    print("ZED相机初始化成功")
    return zed

def capture_stereo_images_zed(zed_camera):
    """
    使用ZED SDK同步捕获左右立体图像，确保两帧严格同步
    
    参数:
        zed_camera (sl.Camera): 初始化后的ZED相机对象
        
    返回:
        tuple: (左相机图像, 右相机图像)，若捕获失败则返回(None, None)
    """
    # 准备图像容器
    left_image = sl.Mat()
    right_image = sl.Mat()
    
    # 抓取一帧（同步采集）
    if zed_camera.grab() == sl.ERROR_CODE.SUCCESS:
        # 获取当前帧的时间戳（可选，用于更严格的同步需求）
        timestamp = zed_camera.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        # 同步检索左右图像
        zed_camera.retrieve_image(left_image, sl.VIEW.LEFT)
        zed_camera.retrieve_image(right_image, sl.VIEW.RIGHT)
        
        # 转换为OpenCV格式（BGR）
        left_cv = left_image.get_data()
        right_cv = right_image.get_data()
        
        return left_cv, right_cv
    else:
        print("错误：无法从ZED相机同步获取图像。")
        return None, None

def detect_drones(image, model):
    """
    使用YOLO模型检测图像中的无人机
    
    参数:
        image (ndarray): 输入图像
        model (YOLO): 加载的YOLO模型
        
    返回:
        ndarray: 检测结果，格式为 [[x, y, width, height], ...]
            x, y: 边界框左上角坐标
            width, height: 边界框宽高
    """
    results = model(image)
    return results.xywh[:, :4]  # 返回x, y, w, h

def triangulate_points(point_left, point_right, P1, P2):
    """
    对匹配的立体图像点进行三角测量，计算3D坐标
    
    参数:
        point_left (tuple): 左图像中的点 (u, v)
        point_right (tuple): 右图像中的点 (u, v)
        P1 (ndarray): 左相机投影矩阵
        P2 (ndarray): 右相机投影矩阵
        
    返回:
        ndarray: 三维点坐标 [X, Y, Z]，以左相机光心为原点，若三角化失败则返回None
    """
    # OpenCV triangulatePoints 需要 (2, N) 格式的点
    pt_left = np.array([point_left], dtype=np.float32).reshape(2, -1)
    pt_right = np.array([point_right], dtype=np.float32).reshape(2, -1)

    # 三角化得到齐次坐标 (4xN)
    points_4d_hom = cv2.triangulatePoints(P1, P2, pt_left, pt_right)
    
    if points_4d_hom is None or points_4d_hom.shape[1] == 0:
        return None
        
    # 转换为非齐次坐标 (X, Y, Z)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)
    
    # 检查Z坐标是否为正（在相机前方）
    if points_3d.shape[0] > 0 and points_3d[0][2] > 0:
        return points_3d[0]  # 返回 [X, Y, Z]
    else:
        return None

def get_sensor_data():
    """
    获取相机平台当前的GPS和IMU数据
    
    返回:
        dict: 包含位置和姿态信息的字典
            latitude (float): 纬度，单位度
            longitude (float): 经度，单位度
            altitude (float): 海拔高度，单位米
            roll (float): 横滚角，单位度
            pitch (float): 俯仰角，单位度
            yaw (float): 偏航角，单位度，相对于正北方向
            
    注意:
        此函数为示例，实际应用中应替换为从实际传感器获取数据的代码
    """
    # 示例数据，应替换为实际的传感器数据获取代码
    return {
        'latitude': 34.0522,
        'longitude': -118.2437,
        'altitude': 71.0,
        'roll': 0.0,
        'pitch': 90.0,  # 朝天为90度
        'yaw': 180.0    # 相机Y轴相对于正北方向的角度
    }

def visualize_detections(frame, detections, drone_info=None):
    """
    在图像上绘制检测框和无人机相关信息
    
    参数:
        frame (ndarray): 输入图像
        detections (ndarray): 检测结果，格式为 [[x_center, y_center, w, h, track_id], ...]
        drone_info (dict, optional): 无人机信息，如距离、GPS位置等
    
    返回:
        ndarray: 绘制完成的图像
    """
    display_frame = frame.copy()
    
    # 绘制检测框
    for detection in detections: # Modified to iterate through detections
        if len(detection) == 5: # Check if track_id is present
            x_center, y_center, w, h, track_id = detection
            label = f'Drone {int(track_id)}'
        elif len(detection) == 4: # Fallback if no track_id (e.g. DeepSORT disabled)
            x_center, y_center, w, h = detection
            label = 'Drone'
        else:
            continue # Skip malformed detections

        # Convert center_x, center_y, w, h to x1, y1 (top-left)
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)
        w = int(w)
        h = int(h)
        
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 如果有额外信息，在图像上显示
    if drone_info:
        if 'distance' in drone_info:
            cv2.putText(display_frame, f"Dist: {drone_info['distance']:.1f}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if 'gps' in drone_info:
            gps = drone_info['gps']
            cv2.putText(display_frame, 
                       f"GPS: {gps['lat']:.4f}, {gps['lon']:.4f}, {gps['alt']:.1f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return display_frame

def main():
    """
    主函数：执行无人机检测与定位的完整流程
    使用ZED2i相机进行图像采集，但不使用其内置深度功能
    """
    # --- 配置参数 ---
    model_path = 'models/yolo11s.pt'
    backup_calibration_file = 'params/stereo_calibration.yaml'  # 备用标定文件
    reid_model_path = 'models/osnet_x0_25_msmt17.pt' # Path for DeepSORT ReID model
    
    # --- 1. 初始化ZED相机（禁用深度功能） ---
    zed = init_zed_camera(resolution=sl.RESOLUTION.HD720, fps=30)
    if zed is None:
        print("无法初始化ZED相机，退出程序。")
        return
    
    # --- 2. 加载标定参数 ---
    # 直接从ZED相机获取标定参数，或从备用文件加载
    calib_params = load_calibration_data(zed_camera=zed, calibration_file=backup_calibration_file)
    if not calib_params: 
        zed.close()
        return
        
    (cam_matrix_left, dist_coeffs_left, cam_matrix_right, dist_coeffs_right, 
     R_stereo, T_stereo, R1, R2, P1, P2, Q, image_size, roi_left, roi_right) = calib_params
    
    # --- 3. 计算立体校正映射（只需一次） ---
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        cam_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        cam_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)
    
    # --- 4. 初始化组件 ---
    # 初始化坐标转换器
    geo_transformer = CoordinateTransformer(sensor_data=get_sensor_data())
    
    # 初始化立体匹配器
    stereo_matcher = StereoMatcher(image_size, P1, P2)
    
    # --- 5. 加载无人机检测模型 ---
    print("Initializing Drone Tracker (YOLO + DeepSORT)...")
    try:
        # detection_model = YOLO(model_path) # Original YOLO model loading
        # Initialize the new DroneTracker
        drone_tracker_left = DroneTracker(yolo_model_path=model_path, reid_model_path=reid_model_path)
        drone_tracker_right = DroneTracker(yolo_model_path=model_path, reid_model_path=reid_model_path)
        print("Drone Tracker initialized.")
    except Exception as e:
        print(f"Drone Tracker initialization failed: {e}")
        zed.close()
        return

    print("开始主循环...")
    try:
        while True:
            start_time = time.time()

            # --- 6. 获取图像 ---
            frame_left_raw, frame_right_raw = capture_stereo_images_zed(zed)
            if frame_left_raw is None or frame_right_raw is None:
                print("无法获取图像，退出循环")
                break

            # --- 7. 立体校正（消除畸变并对齐图像） ---
            # 注意：即使ZED相机提供了校正图像，我们仍然应用我们自己的校正以确保与我们的标定参数一致
            frame_left_rect = cv2.remap(frame_left_raw, map1_left, map2_left, cv2.INTER_LINEAR)
            frame_right_rect = cv2.remap(frame_right_raw, map1_right, map2_right, cv2.INTER_LINEAR)
            
            # 可选：显示校正后的左右图像
            # combined_rect = np.hstack((frame_left_rect, frame_right_rect))
            # cv2.imshow('Rectified Stereo Images', combined_rect)

            # --- 8. 无人机检测 ---
            # detections_left = detect_drones(frame_left_rect, detection_model) 
            # detections_right = detect_drones(frame_right_rect, detection_model)
            
            # Use DroneTracker for detection and tracking
            tracked_objects_left, raw_detections_left = drone_tracker_left.update(frame_left_rect)
            tracked_objects_right, raw_detections_right = drone_tracker_right.update(frame_right_rect)


            # --- 9. 立体匹配 ---
            # Use raw_detections_left (xywh format) for stereo matching, as it's the direct output from YOLO
            # The tracked_objects_left contains track_id, which is useful for visualization
            matched_pairs = stereo_matcher.match(
                raw_detections_left, raw_detections_right, strategy='auto')

            # 用于显示的信息
            drone_info = {}

            # --- 10. 处理匹配结果 ---
            if matched_pairs:
                # 取第一个匹配对进行处理（如果需要处理多个，可以扩展）
                point_left, point_right = matched_pairs[0]
                
                # --- 11. 三维重建（使用传统三角测量，不使用ZED深度） ---
                drone_cam_coords = triangulate_points(point_left, point_right, P1, P2)

                if drone_cam_coords is not None:
                    X, Y, Z = drone_cam_coords
                    # 保存距离信息用于显示
                    drone_info['distance'] = Z
                    
                    # --- 12. 坐标转换（相机系 -> 经纬高） ---
                    drone_gps_coords = geo_transformer(drone_cam_coords)

                    if drone_gps_coords:
                        # 保存GPS信息用于显示
                        drone_info['gps'] = {
                            'lat': drone_gps_coords['lat'],
                            'lon': drone_gps_coords['lon'],
                            'alt': drone_gps_coords['alt']
                        }
                        
                        # 打印结果
                        print(f"检测到无人机 @ 距离: {Z:.2f}m, GPS: "
                              f"Lat={drone_gps_coords['lat']:.6f}, "
                              f"Lon={drone_gps_coords['lon']:.6f}, "
                              f"Alt={drone_gps_coords['alt']:.2f}m")

            # --- 13. 可视化结果 ---
            display_frame = visualize_detections(frame_left_rect, tracked_objects_left, drone_info) # Use tracked_objects_left for visualization
            
            # 添加FPS信息
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.2f}", 
                        (10, image_size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('Drone Detection', display_frame)

            # 检测按键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("用户中断，退出程序")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # --- 14. 释放资源 ---
        print("正在关闭...")
        zed.close()  # 关闭ZED相机
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()