"""
立体匹配模块 - 提供双目图像中检测目标的匹配功能

本模块实现了不同的立体匹配策略，用于在立体相机图像对中找到同一物体的对应点，
以便进行后续的三维重建。
"""

import cv2
import numpy as np

class StereoMatcher:
    """
    处理立体相机匹配的类，支持多种匹配策略
    
    主要功能：
    - 直接中心点匹配：适用于简单场景
    - 基于极线约束的匹配：适用于复杂场景
    - 自动选择匹配策略
    """

    def __init__(self, image_size, P1=None, P2=None, epipolar_threshold=2.0):
        """
        初始化立体匹配器
        
        参数:
            image_size (tuple): 图像尺寸 (width, height)
            P1 (ndarray, optional): 左相机投影矩阵
            P2 (ndarray, optional): 右相机投影矩阵
            epipolar_threshold (float): 极线约束容差（像素），默认2.0
        """
        self.image_size = image_size
        self.P1 = P1
        self.P2 = P2
        self.epipolar_threshold = epipolar_threshold
        
    def match_by_center(self, detections_left, detections_right):
        """
        简单匹配策略：直接匹配左右图像中的单个目标框中心
        
        适用于场景中只有一个目标的简单情况
        
        参数:
            detections_left (ndarray): 左图像检测结果，格式 [[x, y, w, h], ...]
            detections_right (ndarray): 右图像检测结果，格式 [[x, y, w, h], ...]
            
        返回:
            list: 匹配对列表 [(point_left, point_right), ...] 
                 point 是图像像素坐标 (u, v)
        """
        matched_pairs = []
        if not detections_left or not detections_right:
            return matched_pairs

        # 提取检测框中心点
        centers_left = [(x + w/2, y + h/2) for (x, y, w, h) in detections_left]
        centers_right = [(x + w/2, y + h/2) for (x, y, w, h) in detections_right]

        # 简单情况：只有一个检测框，直接匹配
        if len(centers_left) == 1 and len(centers_right) == 1:
            matched_pairs.append((centers_left[0], centers_right[0]))
            
        return matched_pairs
        
    def match_by_epipolar(self, detections_left, detections_right):
        """
        基于极线约束的匹配策略
        
        对于校正后的图像，同一个物体在左右图像中应该有相同的y坐标（水平线约束）
        并且左图中的点x坐标应大于右图中对应点的x坐标（视差为正）
        
        参数:
            detections_left (ndarray): 左图像检测结果，格式 [[x, y, w, h], ...]
            detections_right (ndarray): 右图像检测结果，格式 [[x, y, w, h], ...]
            
        返回:
            list: 匹配对列表 [(point_left, point_right), ...] 
        """
        matched_pairs = []
        if not detections_left or not detections_right:
            return matched_pairs
            
        # 提取检测框中心点
        centers_left = [(x + w/2, y + h/2) for (x, y, w, h) in detections_left]
        centers_right = [(x + w/2, y + h/2) for (x, y, w, h) in detections_right]
        
        # 为左图像中的每个点查找右图像中最佳匹配
        for pl in centers_left:
            best_match = None
            min_dist = float('inf')
            
            # 在极线约束下查找最佳匹配（校正后图像中同一物体的y坐标应相似）
            for pr in centers_right:
                # 检查极线约束（y坐标相似）
                if abs(pl[1] - pr[1]) > self.epipolar_threshold:
                    continue
                
                # 由于是校正后的图像，左图中的点的x坐标应大于右图中对应点的x坐标（视差为正）
                if pl[0] < pr[0]:
                    continue
                    
                # 计算x轴视差（这应该为正）
                disparity = pl[0] - pr[0]
                if disparity <= 0:
                    continue
                    
                # 选择最佳匹配（在满足约束的情况下）
                dist = abs(pl[1] - pr[1])  # y坐标差
                if dist < min_dist:
                    min_dist = dist
                    best_match = pr
            
            # 如果找到匹配点，添加到结果中
            if best_match is not None:
                matched_pairs.append((pl, best_match))
                
        return matched_pairs
    
    def match(self, detections_left, detections_right, strategy='auto'):
        """
        根据指定策略执行立体匹配
        
        参数:
            detections_left (ndarray): 左图像检测结果，格式 [[x, y, w, h], ...]
            detections_right (ndarray): 右图像检测结果，格式 [[x, y, w, h], ...]
            strategy (str): 匹配策略，可选值:
                - 'center': 中心点直接匹配（简单场景）
                - 'epipolar': 基于极线约束的匹配（复杂场景）
                - 'auto': 自动选择策略（默认）
            
        返回:
            list: 匹配对列表 [(point_left, point_right), ...]
        """
        if not detections_left or not detections_right:
            return []
            
        # 根据场景自动选择策略
        if strategy == 'auto':
            if len(detections_left) == 1 and len(detections_right) == 1:
                strategy = 'center'
            else:
                strategy = 'epipolar'
                
        # 执行匹配
        if strategy == 'center':
            return self.match_by_center(detections_left, detections_right)
        elif strategy == 'epipolar':
            return self.match_by_epipolar(detections_left, detections_right)
        else:
            raise ValueError(f"不支持的匹配策略: {strategy}")