import cv2
import numpy as np
import glob
import os
import yaml

# --- 配置参数 ---
CHESSBOARD_SIZE = (10, 7)   # 棋盘格内部角点数量 (必须与 capture 脚本一致)
SQUARE_SIZE = 0.020        # 棋盘格方块的实际边长 (米)。请务必测量并修改此值！

IMAGE_DIR_BASE = "calib_images" # 保存标定图像的基础目录
IMAGE_DIR_LEFT = os.path.join(IMAGE_DIR_BASE, "left")
IMAGE_DIR_RIGHT = os.path.join(IMAGE_DIR_BASE, "right")
OUTPUT_FILE = "stereo_calibration.yaml" # 输出标定结果的文件

# 标定参数设置
# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备世界坐标系中的角点坐标 (0,0,0), (1*square_size,0,0), (2*square_size,0,0) ....,(width-1*square_size,height-1*square_size,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE # 缩放到实际尺寸

# 用于存储所有图像的角点（世界坐标和图像坐标）
objpoints = []  # 存储世界坐标系下的点
imgpoints_left = [] # 存储左相机图像坐标系下的点
imgpoints_right = [] # 存储右相机图像坐标系下的点

# --- 读取图像文件 ---
images_left = sorted(glob.glob(os.path.join(IMAGE_DIR_LEFT, '*.png')))
images_right = sorted(glob.glob(os.path.join(IMAGE_DIR_RIGHT, '*.png')))

if not images_left or not images_right:
    print(f"错误：在 '{IMAGE_DIR_LEFT}' 或 '{IMAGE_DIR_RIGHT}' 中未找到图像。")
    print("请先运行 capture_calibration_images.py 采集图像。")
    exit()

if len(images_left) != len(images_right):
    print(f"警告：左右图像数量不匹配 ({len(images_left)} vs {len(images_right)})。将使用较小的数量。")
    # 确保图像对齐（如果命名规则严格，通常不需要，但以防万一）
    base_names_left = {os.path.basename(f).split('_', 1)[1] for f in images_left}
    base_names_right = {os.path.basename(f).split('_', 1)[1] for f in images_right}
    common_names = sorted(list(base_names_left.intersection(base_names_right)))
    images_left = [os.path.join(IMAGE_DIR_LEFT, f"left_{name}") for name in common_names]
    images_right = [os.path.join(IMAGE_DIR_RIGHT, f"right_{name}") for name in common_names]
    
num_pairs_found = len(images_left)
if num_pairs_found == 0:
     print("错误：找不到匹配的左右图像对。")
     exit()
     
print(f"找到 {num_pairs_found} 对标定图像。开始处理...")

img_shape = None # 用于存储图像尺寸

# --- 查找角点 ---
for i, (fname_left, fname_right) in enumerate(zip(images_left, images_right)):
    img_left = cv2.imread(fname_left)
    img_right = cv2.imread(fname_right)

    if img_left is None or img_right is None:
        print(f"警告：无法读取图像对 {fname_left}, {fname_right}。跳过。")
        continue

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray_left.shape[::-1] # (width, height)
        print(f"图像尺寸检测为: {img_shape}")

    # 查找棋盘格角点
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, flags=flags)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, flags=flags)

    # 如果在两个图像中都找到了角点
    if ret_left and ret_right:
        print(f"在图像对 {i+1}/{num_pairs_found} 中找到角点。")
        objpoints.append(objp)

        # 优化角点位置 (亚像素精度)
        corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        
        imgpoints_left.append(corners_left_subpix)
        imgpoints_right.append(corners_right_subpix)

    else:
        print(f"警告：在图像对 {i+1}/{num_pairs_found} ({os.path.basename(fname_left)}, {os.path.basename(fname_right)}) 中未能同时找到角点。跳过此对。")

cv2.destroyAllWindows()

if not objpoints:
    print("错误：没有找到任何有效的标定图像对。无法进行标定。")
    exit()
    
print(f"\n成功处理了 {len(objpoints)} 对有效图像。开始相机标定...")

# --- 单目标定 (获取初始参数) ---
print("正在进行单目标定 (获取初始值)...")
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
print("左相机单目标定完成。")
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)
print("右相机单目标定完成。")

if not ret_left or not ret_right:
    print("错误：单目标定失败。请检查图像质量和角点检测。")
    exit()
    
# --- 立体标定 ---
print("\n正在进行立体标定...")
# 设置立体标定 flag: 固定内参，使用初始值，尝试更精确的标定
stereo_flags = cv2.CALIB_FIX_INTRINSIC
# stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS
# stereo_flags |= cv2.CALIB_RATIONAL_MODEL # 可选，如果畸变复杂
# stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST

# 注意：传入的是单目标定得到的 mtx 和 dist 作为初始猜测
ret_stereo, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left,   # 使用左相机标定结果作为 M1, d1 输入
    mtx_right, dist_right, # 使用右相机标定结果作为 M2, d2 输入
    img_shape,
    flags=stereo_flags,
    criteria=criteria
)

if ret_stereo:
    print(f"立体标定完成。RMS 重投影误差: {ret_stereo:.4f}")
    print(f"相机矩阵 (左):\n{M1}")
    print(f"畸变系数 (左):\n{d1}")
    print(f"相机矩阵 (右):\n{M2}")
    print(f"畸变系数 (右):\n{d2}")
    print(f"旋转矩阵 R (右相机相对于左相机):\n{R}")
    print(f"平移向量 T (右相机相对于左相机, 米):\n{T}")
    baseline = np.linalg.norm(T)
    print(f"计算基线长度: {baseline:.4f} 米")
    if abs(baseline - np.linalg.norm(T.flatten())) > 1e-6 :
         print("Warning: Baseline calculation might be incorrect if T is not a simple vector.")
         
else:
    print("错误：立体标定失败。")
    exit()

# --- 立体校正 ---
print("\n正在进行立体校正...")
# alpha = 0: 保留所有像素，可能有黑色区域。alpha = 1: 裁剪掉无效像素。
alpha = 0 
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
    M1, d1, M2, d2, img_shape, R, T, alpha=alpha
)

print("立体校正完成。")
print(f"校正旋转矩阵 R1 (左):\n{R1}")
print(f"校正旋转矩阵 R2 (右):\n{R2}")
print(f"投影矩阵 P1 (左):\n{P1}")
print(f"投影矩阵 P2 (右):\n{P2}")
print(f"视差-深度映射矩阵 Q:\n{Q}")
print(f"有效像素区域 ROI Left: {roi_left}")
print(f"有效像素区域 ROI Right: {roi_right}")

# --- 保存结果到 YAML 文件 ---
print(f"\n正在将标定结果保存到 {OUTPUT_FILE}...")

calibration_data = {
    'image_size': list(img_shape), # (width, height)
    'camera_matrix_left': M1.tolist(),
    'dist_coeffs_left': d1.tolist(),
    'camera_matrix_right': M2.tolist(),
    'dist_coeffs_right': d2.tolist(),
    'R': R.tolist(), # Rotation matrix from left to right camera
    'T': T.flatten().tolist(), # Translation vector from left to right camera (in meters)
    'E': E.tolist(), # Essential matrix
    'F': F.tolist(), # Fundamental matrix
    'R1': R1.tolist(), # Rectification transform (rotation matrix) for left camera
    'R2': R2.tolist(), # Rectification transform (rotation matrix) for right camera
    'P1': P1.tolist(), # Projection matrix in the rectified coordinate system for left camera
    'P2': P2.tolist(), # Projection matrix in the rectified coordinate system for right camera
    'Q': Q.tolist(),   # Disparity-to-depth mapping matrix
    'roi_left': list(roi_left), # Valid pixel region in the rectified left image
    'roi_right': list(roi_right), # Valid pixel region in the rectified right image
    'reprojection_error_rms': ret_stereo,
    'square_size_meters': SQUARE_SIZE,
    'num_valid_pairs': len(objpoints)
}

try:
    with open(OUTPUT_FILE, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=None, sort_keys=False)
    print("标定结果成功保存。")
except Exception as e:
    print(f"错误：无法将结果写入 {OUTPUT_FILE}: {e}")
print("\n标定流程结束。")
