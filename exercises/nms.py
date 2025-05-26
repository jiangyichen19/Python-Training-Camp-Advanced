# exercises/nms.py
"""
练习：非极大值抑制 (Non-Maximum Suppression, NMS)

描述：
实现目标检测中常用的 NMS 算法，用于去除重叠度高的冗余边界框。

请补全下面的函数 `calculate_iou` 和 `nms`。
"""
import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比 (IoU)。
    边界框格式：[x_min, y_min, x_max, y_max]
    """
    # 计算相交区域的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # 计算相交区域的面积
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height
    
    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IoU，处理除零情况
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def nms(boxes, scores, iou_threshold):
    """
    执行非极大值抑制 (NMS)。
    """
    # 如果boxes为空，返回空列表
    if len(boxes) == 0:
        return []
    
    # 确保输入是numpy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 计算所有边界框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 根据分数降序排序
    order = np.argsort(scores)[::-1]
    
    keep = []
    while order.size > 0:
        # 保留当前分数最高的边界框
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # 计算当前框与其他框的IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        # 计算IoU
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # 保留IoU小于阈值的框
        inds = np.where(overlap <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep