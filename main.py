import cv2
import numpy as np
import time
import yaml # 用于加载/保存标定数据
from ultralytics import YOLO # 用于无人机检测
from pyproj import Proj, Transformer, CRS # 用于坐标转换
from scipy.spatial.transform import Rotation as R 
from utils.geo_transform import CoordinateTransformer # 假设你有一个 geo_transform.py 文件，包含坐标转换函数

# --- 1. 标定参数加载 ---
def load_calibration_data(calibration_file):
    """从文件加载立体标定和校正参数"""
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
        
        # 计算校正参数 (如果标定脚本没有保存，可以在这里计算并保存)
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            cam_matrix_left, dist_coeffs_left,
            cam_matrix_right, dist_coeffs_right,
            image_size, R, T, alpha=0 )
        roi_left = tuple(calib_data['roi_left'])
        roi_right = tuple(calib_data['roi_right'])

        print("标定参数加载成功。")
        return (cam_matrix_left, dist_coeffs_left, cam_matrix_right, dist_coeffs_right, 
                R, T, R1, R2, P1, P2, Q, image_size, roi_left, roi_right)
                
    except FileNotFoundError:
        print(f"错误：标定文件 {calibration_file} 未找到。请先进行相机标定。")
        exit()
    except Exception as e:
        print(f"加载标定文件时出错: {e}")
        exit()

# --- 2. 图像获取 ---
def capture_stereo_images(cap_left, cap_right):
    """同步获取左右相机图像帧"""
    # 实际应用需要更精确的同步机制 
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        print("错误：无法从一个或两个相机获取图像。")
        return None, None
        
    return frame_left, frame_right

# --- 3. 无人机检测 (占位符) ---
def detect_drones(image, model):
    results = model(image)
    return results.xywh[:, :2]

# --- 4. 立体匹配 (基于检测框中心) ---
def match_stereo_points_from_detections(detections_left, detections_right, P1, P2, image_size):
    """
    根据左右图像的检测结果进行匹配。
    简化方法：假设左右图都能检测到，并基于极线约束和框中心距离进行匹配。
    更鲁棒的方法可能需要特征匹配或更复杂的逻辑。
    输入: 检测框列表 (left, right), 校正后的投影矩阵 P1, P2
    输出: 匹配对列表 [(point_left, point_right), ...] 
          point 是校正后图像的像素坐标 (u, v)
    """
    matched_pairs = []
    if not detections_left or not detections_right:
        return matched_pairs

    # 简单地取每个检测框的中心点
    centers_left = [(x + w / 2, y + h / 2) for (x, y, w, h) in detections_left]
    centers_right = [(x + w / 2, y + h / 2) for (x, y, w, h) in detections_right]

    # --- 简化匹配逻辑 ---
    # 示例：假设只有一个检测，直接配对
    if len(centers_left) == 1 and len(centers_right) == 1:
         # 注意：这里的点是原始图像坐标，如果后续三角化使用校正后的投影矩阵P1, P2，
         # 理论上应该使用校正后图像上的对应点。
         # 但如果检测是在校正后的图像上做的，这里可以直接用。
         # 如果检测在原始图像上做，需要先将点进行校正变换。
         # 为简化，这里假设检测在校正后图像完成，或者直接用原始点配合原始矩阵（更复杂）
         # 我们将在主循环中进行图像校正，所以检测应该在校正后的图像上进行。
         matched_pairs.append((centers_left[0], centers_right[0])) 
         
    # TODO: 实现更鲁棒的匹配算法（例如，对每个左边点，在右边极线上搜索最佳匹配点）
    
    return matched_pairs

# --- 5. 三维重建 ---
def triangulate_points(point_left, point_right, P1, P2):
    """
    使用校正后的投影矩阵 P1, P2 对匹配点进行三角化，得到相机坐标系下的3D坐标。
    输入: point_left (u,v), point_right (u,v) - 必须是校正后图像上的对应点
          P1, P2 - 校正后的投影矩阵 (3x4)
    输出: 3D point (X, Y, Z) in the coordinate system of the left camera, or None if triangulation fails.
    """
    # OpenCV triangulatePoints 需要 (2, N) 或 (N, 1, 2) 格式的点
    pt_left_undist = np.array(point_left, dtype=np.float32).reshape(-1, 1, 2)
    pt_right_undist = np.array(point_right, dtype=np.float32).reshape(-1, 1, 2)

    # 三角化得到齐次坐标 (4xN)
    points_4d_hom = cv2.triangulatePoints(P1, P2, pt_left_undist, pt_right_undist)

    if points_4d_hom is None or points_4d_hom.shape[1] == 0:
        return None
        
    # 转换为非齐次坐标 (X, Y, Z)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)
    
    # 通常只有一个点
    if points_3d.shape[0] > 0:
        # 检查Z坐标是否为正（在相机前方）
        if points_3d[0][2] > 0:
             return points_3d[0] # 返回 [X, Y, Z]
        else:
             # print("Triangulation resulted in point behind camera.")
             return None
    else:
        return None

# --- 6. 传感器数据获取 (占位符) ---
def get_sensor_data():
    """
    获取相机平台当前的 GPS 和 IMU 数据。
    这是一个占位符函数，你需要根据你的硬件接口实现。
    输出: dict {'lat': float, 'lon': float, 'alt': float, 
                 'roll': float, 'pitch': float, 'yaw': float} 
          单位: 纬度/经度 (度), 海拔 (米), 姿态角 (度或弧度，需统一)
    """
    # --- 在这里替换为你的 GPS/IMU 读取代码 ---
    # 示例: 返回固定或随机值
    return {
        'latitude': 34.0522,  # 示例: 洛杉矶市中心纬度
        'lontitude': -118.2437, # 示例: 洛杉矶市中心经度
        'altitude': 71.0,      # 示例: 海拔 (米)
        'roll': 0.0,     # 示例: 横滚角 (度)
        'pitch': 90.0,    # 示例: 俯仰角 (度) - 朝天为90度
        'yaw': 180.0      # 示例: 偏航角 (度) - 相机Y轴相对于正北方向的角度
    }
    # --- GPS/IMU 代码结束 ---

# --- 主函数 ---
def main():
    calibration_file = 'params/stereo_calibration.yaml' # 标定结果保存的文件
    model_path = 'models/yolo11s.pt' # 或 .pt, .pb 等

    # 1. 加载标定参数
    calib_params = load_calibration_data(calibration_file)
    if not calib_params: return
    (cam_matrix_left, dist_coeffs_left, cam_matrix_right, dist_coeffs_right, 
     R_stereo, T_stereo, R1, R2, P1, P2, Q, image_size, roi_left, roi_right) = calib_params
    geo_transformer = CoordinateTransformer(sensor_data=get_sensor_data()) # 假设你有一个坐标转换器类
    # 2. 初始化相机
    cap_left = cv2.VideoCapture(0) # 调整索引
    cap_right = cv2.VideoCapture(1) # 调整索引

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("错误：无法打开一个或两个相机。")
        return

    # 3. 加载无人机检测模型 (占位符)
    print("加载无人机检测模型... (占位符)")
    # detection_model = load_my_model(model_path) 
    detection_model = YOLO(model_path) # 替换为实际模型加载
    print("模型加载完成。")

    # 4. 计算立体校正映射 (只需一次)
    map1_left, map2_left = cv2.initUndistortRectifyMap(cam_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(cam_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

    print("开始主循环...")
    while True:
        start_time = time.time()

        # 5. 获取图像
        frame_left_raw, frame_right_raw = capture_stereo_images(cap_left, cap_right)
        if frame_left_raw is None: break

        # 6. 立体校正 (消除畸变并将图像对准)
        frame_left_rect = cv2.remap(frame_left_raw, map1_left, map2_left, cv2.INTER_LINEAR)
        frame_right_rect = cv2.remap(frame_right_raw, map1_right, map2_right, cv2.INTER_LINEAR)
        
        # --- 可视化校正后的图像 ---
        # combined_rect = np.hstack((frame_left_rect, frame_right_rect))
        # cv2.imshow('Rectified Stereo Images', combined_rect)

        # 7. 无人机检测 (在校正后的左图或双图进行)
        detections_left = detect_drones(frame_left_rect, detection_model) 
        detections_right = detect_drones(frame_right_rect, detection_model)

        # --- 可视化检测结果 ---
        display_frame = frame_left_rect.copy()
        for (x, y, w, h) in detections_left:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, 'Drone', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 8. 立体匹配
        # 注意：输入给 match 函数的应该是校正后图像上的检测结果/点
        matched_pairs = match_stereo_points_from_detections(detections_left, detections_right, P1, P2, image_size)

        # 9. 对每个匹配对进行处理
        for point_left, point_right in matched_pairs:
            # 10. 三维重建 (得到相机坐标系下的 XYZ)
            drone_cam_coords = triangulate_points(point_left, point_right, P1, P2)

            if drone_cam_coords is not None:
                X, Y, Z = drone_cam_coords
                # print(f"无人机相机坐标 (左相机系): X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m")
                cv2.putText(display_frame, f"Dist: {Z:.1f}m", (int(point_left[0]), int(point_left[1]) - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 12. 坐标转换 (相机系 -> 经纬高)
                drone_gps_coords = geo_transformer(drone_cam_coords)

                if drone_gps_coords:
                    lat = drone_gps_coords['lat']
                    lon = drone_gps_coords['lon']
                    alt = drone_gps_coords['alt']
                    print(f"检测到无人机 @ GPS: Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt:.2f}m")
                    cv2.putText(display_frame, f"GPS: {lat:.4f}, {lon:.4f}, {alt:.1f}m", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 13. 显示结果 & 帧率
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, image_size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Drone Detection', display_frame)

        # 14. 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 15. 释放资源
    print("正在关闭...")
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 并将结果 (cam_matrix_left, dist_coeffs_left, ..., P1, P2, Q, R1, R2, roi_left, roi_right) 保存到 yaml 文件中
    main()