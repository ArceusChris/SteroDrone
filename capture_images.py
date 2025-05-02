import cv2
import time
import os
import numpy as np

# --- 配置参数 ---
CHESSBOARD_SIZE = (10, 7)  # 棋盘格内部角点数量 (列数-1, 行数-1)
NUM_PAIRS_TARGET = 20      # 目标采集的有效图像对数量
CAPTURE_DELAY = 2.0        # 每次尝试捕获之间的延迟（秒）
SAVE_DIR_BASE = "calib_images" # 基础保存目录
SAVE_DIR_LEFT = os.path.join(SAVE_DIR_BASE, "left")
SAVE_DIR_RIGHT = os.path.join(SAVE_DIR_BASE, "right")

# --- 创建保存目录 ---
os.makedirs(SAVE_DIR_LEFT, exist_ok=True)
os.makedirs(SAVE_DIR_RIGHT, exist_ok=True)

# --- 初始化相机 ---
# 尝试不同的索引，直到找到正确的左右相机
cap_left = cv2.VideoCapture(0) 
cap_right = cv2.VideoCapture(1) 

if not cap_left.isOpened():
    print("错误：无法打开左相机 (索引 0)。请检查连接或尝试其他索引。")
    exit()
if not cap_right.isOpened():
    print("错误：无法打开右相机 (索引 1)。请检查连接或尝试其他索引。")
    exit()

print("相机已打开。")

# --- 采集循环 ---
pair_count = 0
last_capture_time = time.time()

print(f"准备采集 {NUM_PAIRS_TARGET} 对标定图像。")
print(f"请将 {CHESSBOARD_SIZE[0]+1}x{CHESSBOARD_SIZE[1]+1} 的棋盘格放置在两个相机视野中。")
print("脚本将每隔 {:.1f} 秒尝试检测并保存一次。按 'q' 退出。".format(CAPTURE_DELAY))

while pair_count < NUM_PAIRS_TARGET:
    # 读取帧
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("错误：无法从相机读取帧。")
        break

    current_time = time.time()
    display_left = frame_left.copy()
    display_right = frame_right.copy()

    # 检查是否到达捕获时间
    if current_time - last_capture_time >= CAPTURE_DELAY:
        last_capture_time = current_time
        print(f"\n尝试捕获第 {pair_count + 1}/{NUM_PAIRS_TARGET} 对图像...")

        # 转换为灰度图
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        # flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_corners_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, flags=flags)
        ret_corners_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, flags=flags)

        # 如果在两个图像中都找到了角点
        if ret_corners_left and ret_corners_right:
            pair_count += 1
            
            # 优化角点位置 (亚像素精度)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            # 保存图像对 (使用原始彩色图像)
            img_name_left = os.path.join(SAVE_DIR_LEFT, f"left_{pair_count:02d}.png")
            img_name_right = os.path.join(SAVE_DIR_RIGHT, f"right_{pair_count:02d}.png")
            cv2.imwrite(img_name_left, frame_left)
            cv2.imwrite(img_name_right, frame_right)
            print(f"成功检测并保存: {img_name_left}, {img_name_right}")

            # 在显示图像上绘制角点 (可选)
            cv2.drawChessboardCorners(display_left, CHESSBOARD_SIZE, corners_left_subpix, ret_corners_left)
            cv2.drawChessboardCorners(display_right, CHESSBOARD_SIZE, corners_right_subpix, ret_corners_right)
        
        elif not ret_corners_left:
             print("未在左图中检测到角点。请调整棋盘格位置或光照。")
        elif not ret_corners_right:
             print("未在右图中检测到角点。请调整棋盘格位置或光照。")


    # 显示状态信息
    text = f"Captured: {pair_count}/{NUM_PAIRS_TARGET}"
    cv2.putText(display_left, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_right, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 显示图像
    cv2.imshow('Left Camera - Press Q to Quit', display_left)
    cv2.imshow('Right Camera - Press Q to Quit', display_right)

    # 检查退出键
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("用户中断采集。")
        break

# --- 清理 ---
print("\n采集结束。")
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print(f"图像已保存到 '{SAVE_DIR_LEFT}' 和 '{SAVE_DIR_RIGHT}' 目录下。")