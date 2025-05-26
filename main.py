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
            # 从ZED相机获取标定参数 - 修正的版本
            camera_info = zed_camera.get_camera_information()
            calibration_params = camera_info.camera_configuration.calibration_parameters
            
            # 左相机参数
            cam_matrix_left = np.array([
                [calibration_params.left_cam.fx, 0, calibration_params.left_cam.cx],
                [0, calibration_params.left_cam.fy, calibration_params.left_cam.cy],
                [0, 0, 1]
            ])
            
            # ZED相机镜头畸变系数
            dist_coeffs_left = np.array([
                calibration_params.left_cam.disto[0],
                calibration_params.left_cam.disto[1],
                calibration_params.left_cam.disto[2],
                calibration_params.left_cam.disto[3],
                calibration_params.left_cam.disto[4]
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
                calibration_params.right_cam.disto[4]
            ])            # 修正：正确获取相机间的转换参数
            # ZED SDK中的立体参数获取方式 - 修正baseline获取方法
            try:
                # 尝试直接获取baseline
                baseline = calibration_params.getCameraBaseline()
                print(f"从ZED SDK获取baseline: {baseline}")
            except AttributeError:
                try:
                    # 备用方式1：从T向量获取
                    if hasattr(calibration_params, 'T'):
                        baseline = abs(calibration_params.T[0])
                        print(f"从T向量获取baseline: {baseline}")
                    else:
                        raise AttributeError("No T vector found")
                except:
                    try:
                        # 备用方式2：从双目标定参数计算
                        cx_diff = abs(calibration_params.left_cam.cx - calibration_params.right_cam.cx)
                        if cx_diff > 0:
                            baseline = cx_diff / calibration_params.left_cam.fx
                            print(f"使用计算方式获取baseline: {baseline}")
                        else:
                            raise ValueError("Invalid cx difference")
                    except:
                        # 默认baseline（ZED2i典型值约120mm = 0.12m）
                        baseline = 0.12
                        print(f"使用默认baseline: {baseline}")
            
            # 验证baseline的有效性
            if baseline <= 0.0 or baseline > 1.0:  # baseline应该在0-1米范围内
                print(f"警告：baseline值异常 ({baseline})，使用默认值")
                baseline = 0.12
            
            # 构造旋转和平移矩阵（ZED相机通常是水平对齐的）
            R = np.eye(3)  # 对于水平对齐的立体相机，旋转矩阵通常是单位矩阵
            T = np.array([-baseline, 0, 0])  # 平移向量，基线距离
            
            # 获取图像尺寸
            camera_resolution = camera_info.camera_configuration.resolution
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
            print("尝试使用备用标定文件...")
    
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
        # 同步检索左右图像
        zed_camera.retrieve_image(left_image, sl.VIEW.LEFT)
        zed_camera.retrieve_image(right_image, sl.VIEW.RIGHT)
        
        # 转换为OpenCV格式（BGR）
        left_cv = left_image.get_data()
        right_cv = right_image.get_data()
        
        # 检查图像是否有效
        if left_cv is None or right_cv is None:
            print("错误：获取的图像数据为空")
            return None, None
            
        # 检查图像形状
        if len(left_cv.shape) != 3 or len(right_cv.shape) != 3:
            print(f"错误：图像维度异常 - Left: {left_cv.shape}, Right: {right_cv.shape}")
            return None, None
            
        # 确保图像是BGR格式（3通道）
        if left_cv.shape[2] == 4:  # BGRA -> BGR
            left_cv = cv2.cvtColor(left_cv, cv2.COLOR_BGRA2BGR)
        if right_cv.shape[2] == 4:  # BGRA -> BGR  
            right_cv = cv2.cvtColor(right_cv, cv2.COLOR_BGRA2BGR)
        
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
    """    # --- 配置参数 ---
    model_path = 'models/yolo11s.pt'
    backup_calibration_file = 'params/stereo_calibration.yaml'  # 备用标定文件
    
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
    print("计算立体校正映射...")
    try:
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            cam_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            cam_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)
        print("立体校正映射计算成功")
        
        # 验证映射是否有效
        if map1_left is None or map2_left is None or map1_right is None or map2_right is None:
            print("错误：立体校正映射为空")
            zed.close()
            return
            
        print(f"映射形状 - Left: {map1_left.shape}, Right: {map1_right.shape}")
        
    except Exception as e:
        print(f"计算立体校正映射失败: {e}")
        zed.close()
        return
      # --- 4. 初始化组件 ---
    # 获取传感器数据
    sensor_data = get_sensor_data()
    
    # 初始化坐标转换器 - 修正参数名称
    camera_extrinsics = {
        'latitude': sensor_data['latitude'],
        'longitude': sensor_data['longitude'],
        'altitude': sensor_data['altitude'],
        'roll': sensor_data['roll'],
        'pitch': sensor_data['pitch'],
        'yaw': sensor_data['yaw']
    }
    geo_transformer = CoordinateTransformer(camera_extrinsics=camera_extrinsics)
    
    # 初始化立体匹配器
    stereo_matcher = StereoMatcher(image_size, P1, P2)
      # --- 5. 加载无人机检测模型 ---
    print("Initializing Drone Tracker (YOLO + ByteTrack)...")
    try:
        # Initialize the new DroneTracker (ByteTrack doesn't need reid_model_path)
        drone_tracker_left = DroneTracker(yolo_model_path=model_path)
        drone_tracker_right = DroneTracker(yolo_model_path=model_path)
        print("Drone Tracker initialized.")
    except Exception as e:
        print(f"Drone Tracker initialization failed: {e}")
        zed.close()
        return    print("开始主循环...")
    try:
        frame_count = 0
        while True:
            start_time = time.time()
            frame_count += 1

            # --- 6. 获取图像 ---
            frame_left_raw, frame_right_raw = capture_stereo_images_zed(zed)
            if frame_left_raw is None or frame_right_raw is None:
                print("无法获取图像，退出循环")
                break
                
            print(f"Frame {frame_count}: 图像获取成功 - Left: {frame_left_raw.shape}, Right: {frame_right_raw.shape}")

            try:
                # --- 7. 立体校正（消除畸变并对齐图像） ---
                print("开始立体校正...")
                frame_left_rect = cv2.remap(frame_left_raw, map1_left, map2_left, cv2.INTER_LINEAR)
                frame_right_rect = cv2.remap(frame_right_raw, map1_right, map2_right, cv2.INTER_LINEAR)
                print("立体校正完成")
                
                # 可选：显示校正后的左右图像
                # combined_rect = np.hstack((frame_left_rect, frame_right_rect))
                # cv2.imshow('Rectified Stereo Images', combined_rect)

                # --- 8. 无人机检测 ---
                print("开始无人机检测...")
                tracked_objects_left, raw_detections_left = drone_tracker_left.update(frame_left_rect)
                tracked_objects_right, raw_detections_right = drone_tracker_right.update(frame_right_rect)
                print(f"检测完成 - Left: {len(raw_detections_left)} detections, Right: {len(raw_detections_right)} detections")

                # --- 9. 立体匹配 ---
                matched_pairs = stereo_matcher.match(
                    raw_detections_left, raw_detections_right, strategy='auto')

                # 用于显示的信息
                drone_info = {}

                # --- 10. 处理匹配结果 ---
                if matched_pairs:
                    print(f"找到 {len(matched_pairs)} 个匹配对")
                    # 取第一个匹配对进行处理（如果需要处理多个，可以扩展）
                    point_left, point_right = matched_pairs[0]
                    
                    # --- 11. 三维重建（使用传统三角测量，不使用ZED深度） ---
                    drone_cam_coords = triangulate_points(point_left, point_right, P1, P2)

                    if drone_cam_coords is not None:
                        X, Y, Z = drone_cam_coords
                        # 保存距离信息用于显示
                        drone_info['distance'] = Z
                          # --- 12. 坐标转换（相机系 -> 经纬高） ---
                        try:
                            # 使用 camera_to_geographic 方法进行坐标转换
                            obj_lat, obj_lon = geo_transformer.camera_to_geographic(drone_cam_coords)
                            
                            # 计算无人机高度（相机高度 + 相对高度）
                            camera_alt = sensor_data['altitude']
                            drone_alt = camera_alt + Y  # Y是相机坐标系中的垂直分量
                            
                            # 保存GPS信息用于显示
                            drone_info['gps'] = {
                                'lat': obj_lat,
                                'lon': obj_lon,
                                'alt': drone_alt
                            }
                            
                            # 打印结果
                            print(f"检测到无人机 @ 距离: {Z:.2f}m, GPS: "
                                  f"Lat={obj_lat:.6f}, "
                                  f"Lon={obj_lon:.6f}, "
                                  f"Alt={drone_alt:.2f}m")
                        except Exception as e:
                            print(f"坐标转换失败: {e}")
                            drone_info['gps'] = None

                # --- 13. 可视化结果 ---
                print("开始可视化...")
                display_frame = visualize_detections(frame_left_rect, tracked_objects_left, drone_info)
                
                # 添加FPS信息
                end_time = time.time()
                fps = 1.0 / (end_time - start_time)
                cv2.putText(display_frame, f"FPS: {fps:.2f}", 
                            (10, image_size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow('Drone Detection', display_frame)
                print(f"Frame {frame_count} 处理完成")

                # 检测按键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as inner_e:
                print(f"处理帧 {frame_count} 时发生错误: {inner_e}")
                print(f"错误类型: {type(inner_e).__name__}")
                import traceback
                traceback.print_exc()
                # 继续处理下一帧而不是退出
                continue
                
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