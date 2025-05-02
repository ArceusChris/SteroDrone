import cv2
import numpy as np
import time
import yaml # 用于加载/保存标定数据
from pyproj import Proj, Transformer, CRS # 用于坐标转换
from scipy.spatial.transform import Rotation as R 

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
    # 临时返回一个假数据用于测试流程
    h, w = image.shape[:2]
    # 模拟在图像中心附近检测到一个 50x50 的框
    if np.random.rand() > 0.5: # 随机模拟检测到与否
         center_x, center_y = w // 2 + np.random.randint(-50, 50), h // 2 + np.random.randint(-50, 50)
         box_w, box_h = 50, 50
         x = max(0, center_x - box_w // 2)
         y = max(0, center_y - box_h // 2)
         return [(x, y, box_w, box_h)]
    else:
         return []
    # --- 模型推理代码结束 ---

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
        'lat': 34.0522,  # 示例: 洛杉矶市中心纬度
        'lon': -118.2437, # 示例: 洛杉矶市中心经度
        'alt': 71.0,      # 示例: 海拔 (米)
        'roll': 0.0,     # 示例: 横滚角 (度)
        'pitch': 90.0,    # 示例: 俯仰角 (度) - 朝天为90度
        'yaw': 45.0      # 示例: 偏航角 (度) - 相机Y轴相对于正北方向的角度
    }
    # --- GPS/IMU 代码结束 ---

# --- 7. 坐标系转换 ---
def transform_to_gps(drone_cam_coords, sensor_data):
    """
    将无人机在相机坐标系下的坐标 (X, Y, Z) 转换到大地坐标系 (经纬高)。
    输入: drone_cam_coords (np.array [X, Y, Z]) - 相机坐标系下的坐标 (米)
          sensor_data (dict) - 包含相机平台 GPS 和 IMU 信息的字典
    输出: dict {'lat': float, 'lon': float, 'alt': float} or None
    """
    try:
        X, Y, Z = drone_cam_coords
        cam_lat = sensor_data['lat']
        cam_lon = sensor_data['lon']
        cam_alt = sensor_data['alt']
        # 确保角度是弧度
        roll = np.radians(sensor_data['roll'])
        pitch = np.radians(sensor_data['pitch'])
        yaw = np.radians(sensor_data['yaw']) # 相对于正北方向

        # 1. 定义坐标系
        # WGS84 Geodetic (lat/lon/alt)
        crs_geodetic = CRS("EPSG:4326") # 经纬度
        crs_geodetic_alt = CRS("EPSG:4979") # 经纬度+海拔 (WGS84 3D)
        # ECEF (Earth-Centered, Earth-Fixed)
        crs_ecef = CRS("EPSG:4978")
        
        # Transformer: Geodetic -> ECEF
        transformer_geo_to_ecef = Transformer.from_crs(crs_geodetic_alt, crs_ecef, always_xy=True)
        # Transformer: ECEF -> Geodetic
        transformer_ecef_to_geo = Transformer.from_crs(crs_ecef, crs_geodetic_alt, always_xy=True)

        # 2. 计算相机在 ECEF 中的位置
        cam_ecef_x, cam_ecef_y, cam_ecef_z = transformer_geo_to_ecef.transform(cam_lon, cam_lat, cam_alt)
        cam_ecef_pos = np.array([cam_ecef_x, cam_ecef_y, cam_ecef_z])

        # 3. 计算从相机坐标系到 ENU (East-North-Up) 或 NED (North-East-Down) 的旋转矩阵
        # 注意：旋转顺序和定义很重要 (e.g., ZYX, XYZ)。这里假设是常见的航空 ZYX 顺序 (Yaw, Pitch, Roll)
        # 并且假设相机坐标系：Z轴沿光轴向前，Y轴向下，X轴向右
        # 如果相机竖直朝天 (pitch=90)，这个标准航空定义可能需要调整，
        # 或者直接定义相机坐标系到本地水平坐标系 (ENU/NED) 的旋转。

        # 假设一个更直接的定义：
        # 相机坐标系 (Cam): Z朝天, Y朝前(某个方向), X朝右 (相对于Y)
        # 本地水平坐标系 (ENU): X (East), Y (North), Z (Up)
        # 需要构建从 Cam 到 ENU 的旋转矩阵 R_enu_from_cam
        
        # --- 这部分旋转矩阵的构建非常关键，需要根据你的具体相机安装和IMU输出定义 ---
        # 示例：假设相机Z轴严格朝上，相机Y轴指向地理北(Yaw=0)，相机X轴指向地理东(Roll=0)
        # 这是一个理想情况，实际需要用IMU的 roll, pitch, yaw 来构建精确的旋转矩阵
        # from scipy.spatial.transform import Rotation as R
        # r = R.from_euler('zyx', [yaw, pitch, roll], degrees=False) 
        # R_ned_from_body = r.as_matrix() # 假设IMU输出的是载具(body)到NED的旋转
        # R_body_from_cam = ... # 需要知道相机如何安装在载具上
        # R_ned_from_cam = R_ned_from_body @ R_body_from_cam
        
        # 简化示例：假设已知 Cam 到 ENU 的旋转 R_enu_from_cam (需要根据实际情况计算)
        # 例如，如果相机 Z 指向 Up，Y 指向 North，X 指向 East (理想朝天)
        # R_enu_from_cam = np.identity(3) 
        # 如果相机 Z 指向 Up，Y 指向 East，X 指向 North
        # R_enu_from_cam = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) 
        
        # 使用 scipy.spatial.transform.Rotation (推荐)
        from scipy.spatial.transform import Rotation as R

        # 假设 IMU 提供了从 NED 到 Body(载具) 的旋转四元数或欧拉角
        # 假设相机安装在载具上，其坐标系 (Cam) 相对于载具坐标系 (Body) 的旋转是 R_body_from_cam
        # 例如：相机光轴(Z)对准载具Z轴，相机Y轴对准载具Y轴，相机X轴对准载具X轴
        # R_body_from_cam = np.identity(3) 
        # 如果相机光轴(Z)对准载具X轴, 相机Y轴对准载具-Z轴, 相机X轴对准载具Y轴
        # R_body_from_cam = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]]).T # .T because we want body_from_cam

        # 假设我们有从 Body 到 NED 的旋转矩阵 R_ned_from_body (由IMU的roll,pitch,yaw计算得到)
        # roll, pitch, yaw 单位是弧度
        r_imu = R.from_euler('zyx', [yaw, pitch, roll]) # 注意欧拉角顺序
        R_ned_from_body = r_imu.as_matrix()

        # 假设相机就是载具 (Body == Cam) 且 Z 轴朝天 (Pitch=pi/2), Y轴朝北 (Yaw=0)
        # 这只是一个非常简化的例子
        R_enu_from_cam = np.array([ # Cam (X右, Y前, Z上) -> ENU (X东, Y北, Z上)
             [1, 0, 0], # ENU East = Cam X
             [0, 1, 0], # ENU North = Cam Y
             [0, 0, 1]  # ENU Up = Cam Z
        ])
        # *** 实际应用中必须根据IMU和相机安装精确计算此旋转矩阵 ***
        
        # 4. 将无人机在相机坐标系的位置向量旋转到 ENU 坐标系
        drone_cam_vec = np.array([X, Y, Z])
        drone_enu_vec = R_enu_from_cam @ drone_cam_vec
        delta_east, delta_north, delta_up = drone_enu_vec

        # 5. 计算无人机在 ECEF 中的位置
        # 近似方法 (适用于短距离): 使用局部ENU坐标系原点(相机位置)的转换矩阵
        # 更精确的方法：直接在ECEF中应用位移，但需要考虑从ENU到ECEF的旋转（取决于纬度）
        
        # 使用 pyproj 的近似方法 (将 ENU 位移添加到相机位置)
        # 定义一个以相机位置为中心的局部坐标系 (例如 Transverse Mercator 或 LCC)
        # 或者，直接使用 ECEF 进行矢量相加（需要将 ENU 向量转换到 ECEF 方向）
        
        # 简化的近似：直接将 ENU 位移加到相机经纬高上 (误差较大，尤其纬度)
        # 不推荐，但作为概念演示：
        # meters_per_degree_lat = 111132.954 - 559.822 * np.cos(2 * np.radians(cam_lat)) + 1.175 * np.cos(4 * np.radians(cam_lat))
        # meters_per_degree_lon = 111412.84 * np.cos(np.radians(cam_lat)) - 93.5 * np.cos(3 * np.radians(cam_lat))
        # drone_lat = cam_lat + delta_north / meters_per_degree_lat
        # drone_lon = cam_lon + delta_east / meters_per_degree_lon
        # drone_alt = cam_alt + delta_up
        
        # 推荐：使用 pyproj 进行精确转换 (涉及到局部坐标系或ECEF旋转)
        # 步骤：Cam Geodetic -> Cam ECEF -> 计算局部ENU基向量在ECEF中的表示 ->
        #       -> 将 drone_enu_vec 转换到 ECEF 偏移 -> Cam ECEF + Offset ECEF = Drone ECEF ->
        #       -> Drone ECEF -> Drone Geodetic
        
        # pyproj 提供了一个更直接的方式（内部处理了局部转换）
        # 定义一个基于相机位置的局部 Azimuthal Equidistant (AEQD) 坐标系
        aeqd_crs = CRS(proj='aeqd', lat_0=cam_lat, lon_0=cam_lon, datum='WGS84', units='m')
        # Transformer: ENU (在AEQD中近似为x,y,z) -> Geodetic
        transformer_local_to_geo = Transformer.from_crs(aeqd_crs, crs_geodetic_alt, always_xy=True)
        
        # 转换 ENU 坐标 (delta_east, delta_north, delta_up) 到目标经纬高
        # 注意：AEQD 的 x, y 通常对应 easting, northing
        drone_lon, drone_lat, drone_alt_relative = transformer_local_to_geo.transform(delta_east, delta_north, delta_up)
        drone_alt = cam_alt + delta_up # AEQD 通常不直接处理高程，我们手动加上
        
        return {'lat': drone_lat, 'lon': drone_lon, 'alt': drone_alt}

    except Exception as e:
        print(f"坐标转换出错: {e}")
        return None

# --- 主函数 ---
def main():
    calibration_file = 'stereo_calibration.yaml' # 标定结果保存的文件
    # model_path = 'path/to/your/drone_detection_model.onnx' # 或 .pt, .pb 等

    # 1. 加载标定参数
    calib_params = load_calibration_data(calibration_file)
    if not calib_params: return
    (cam_matrix_left, dist_coeffs_left, cam_matrix_right, dist_coeffs_right, 
     R_stereo, T_stereo, R1, R2, P1, P2, Q, image_size, roi_left, roi_right) = calib_params

    # 2. 初始化相机
    cap_left = cv2.VideoCapture(0) # 调整索引
    cap_right = cv2.VideoCapture(1) # 调整索引

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("错误：无法打开一个或两个相机。")
        return

    # 3. 加载无人机检测模型 (占位符)
    print("加载无人机检测模型... (占位符)")
    # detection_model = load_my_model(model_path) 
    detection_model = None # 替换为实际模型加载
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

                # 11. 获取传感器数据 (GPS, IMU)
                sensor_data = get_sensor_data()
                if not sensor_data:
                    print("警告：无法获取传感器数据。")
                    continue

                # 12. 坐标转换 (相机系 -> 经纬高)
                drone_gps_coords = transform_to_gps(drone_cam_coords, sensor_data)

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