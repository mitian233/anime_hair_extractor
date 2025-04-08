"""
潘通颜色数据模块 - 提供潘通色卡颜色信息和匹配功能
"""
import numpy as np
from skimage import color
import math
import csv
import os

def load_pantone_colors():
    """
    从CSV文件加载潘通颜色数据
    
    Returns:
        list: 潘通颜色数据列表 [[LAB值, 颜色名称], ...]
    """
    colors = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lab_file = os.path.join(current_dir, 'pantone_lab.csv')
    
    # 读取LAB值
    lab_data = {}
    with open(lab_file, 'r', encoding='utf-8-sig') as f:  # 使用 utf-8-sig 来处理 BOM
        reader = csv.DictReader(f)
        for row in reader:
            name = row['pantone_name']  # 现在不需要处理 BOM 了
            lab_data[name] = [
                float(row['L']),
                float(row['A']),
                float(row['B'])
            ]
            # 添加到颜色列表中，添加"潘通"前缀
            colors.append([lab_data[name], f"潘通 {name}"])

    # 添加动漫常见的特殊发色
    special_colors = [
        [[48, 26, -58], "潘通 薰衣草蓝"],
        [[54, 85, -23], "潘通 品红"],
        [[46, 77, -77], "潘通 紫罗兰"],
        [[83, -40, -5], "潘通 青绿"],
        [[86, 5, 82], "潘通 金色"],
        [[77, 0, 0], "潘通 银色"],
        [[54, -3, -7], "潘通 石板灰"],
        [[66, 15, 12], "潘通 浅玫瑰棕"],
        [[75, 8, 30], "潘通 棕褐色"]
    ]
    colors.extend(special_colors)
    
    return colors

# 加载潘通颜色数据
PANTONE_COLORS = load_pantone_colors()

def rgb_to_lab(rgb_color):
    """
    将RGB颜色转换为CIE Lab颜色空间
    
    Args:
        rgb_color: RGB颜色值 [r, g, b] (0-255)
        
    Returns:
        ndarray: Lab颜色值 [L, a, b]
    """
    # 归一化RGB值到0-1范围
    rgb_normalized = np.array(rgb_color) / 255.0
    # 重塑数组为3D形状
    rgb_reshaped = rgb_normalized.reshape(1, 1, 3)
    # 转换为Lab空间
    lab_color = color.rgb2lab(rgb_reshaped)
    # 返回一维数组 [L, a, b]
    return lab_color[0, 0]

def lab_to_rgb(lab_color):
    """
    将LAB颜色转换为RGB颜色空间
    
    Args:
        lab_color: LAB颜色值 [L, a, b]
        
    Returns:
        ndarray: RGB颜色值 [r, g, b] (0-255)
    """
    # 将LAB值转换为形状 (1, 1, 3) 的数组
    lab_reshaped = np.array([[[lab_color[0], lab_color[1], lab_color[2]]]]) 
    # 转换为RGB
    rgb = color.lab2rgb(lab_reshaped)
    # 转换为0-255范围并返回
    return (rgb[0, 0] * 255).astype(np.uint8)

def color_distance_rgb(color1, color2):
    """
    计算RGB空间中两个颜色的欧氏距离
    
    Args:
        color1: 第一个RGB颜色 [r, g, b]
        color2: 第二个RGB颜色 [r, g, b]
        
    Returns:
        float: 颜色距离
    """
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

def color_distance_lab(color1, color2):
    """
    计算CIE Lab空间中两个颜色的欧氏距离（Delta E）
    
    Args:
        color1: 第一个RGB颜色 [r, g, b]
        color2: 第二个RGB颜色 [r, g, b]
        
    Returns:
        float: 颜色距离
    """
    lab1 = rgb_to_lab(color1)
    lab2 = rgb_to_lab(color2)
    return np.sqrt(np.sum((lab1 - lab2) ** 2))

def delta_e_cie2000(lab1, lab2):
    """
    计算CIE2000标准下的Delta E颜色差异
    
    Args:
        lab1: 第一个Lab颜色 [L, a, b]
        lab2: 第二个Lab颜色 [L, a, b]
        
    Returns:
        float: Delta E 2000值
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # CIE2000算法参数
    kL, kC, kH = 1, 1, 1
    
    # 亮度差异
    L_bar = (L1 + L2) / 2  # 添加这行
    delta_L_prime = L2 - L1
    
    # 色度计算
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    
    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    C_bar_prime = (C1_prime + C2_prime) / 2
    
    delta_C_prime = C2_prime - C1_prime
    
    # 色相计算
    h1_prime = math.atan2(b1, a1_prime) % (2 * math.pi)
    h2_prime = math.atan2(b2, a2_prime) % (2 * math.pi)
    
    delta_h_prime = h2_prime - h1_prime
    if abs(delta_h_prime) > math.pi:
        if h2_prime <= h1_prime:
            delta_h_prime += 2 * math.pi
        else:
            delta_h_prime -= 2 * math.pi
    
    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(delta_h_prime / 2)
    
    # 加权因子
    H_bar_prime = (h1_prime + h2_prime) / 2
    if abs(h1_prime - h2_prime) > math.pi:
        if h1_prime + h2_prime < 2 * math.pi:
            H_bar_prime += math.pi
        else:
            H_bar_prime -= math.pi
    
    T = (1 - 0.17 * math.cos(H_bar_prime - math.pi/6) 
         + 0.24 * math.cos(2 * H_bar_prime) 
         + 0.32 * math.cos(3 * H_bar_prime + math.pi/30) 
         - 0.20 * math.cos(4 * H_bar_prime - 21*math.pi/60))
    
    SL = 1 + (0.015 * (L_bar - 50)**2) / math.sqrt(20 + (L_bar - 50)**2)
    SC = 1 + 0.045 * C_bar_prime
    SH = 1 + 0.015 * C_bar_prime * T
    
    RT = -2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7)) * math.sin(60 * math.exp(-((H_bar_prime - 275*math.pi/180) / 25*math.pi/180)**2))
    
    # 计算总的颜色差异
    delta_E = math.sqrt(
        (delta_L_prime / (kL * SL))**2 +
        (delta_C_prime / (kC * SC))**2 +
        (delta_H_prime / (kH * SH))**2 +
        RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )
    
    return delta_E

def find_closest_pantone(color_value, method='cie2000', input_space='rgb'):
    """
    找出最接近给定颜色的潘通色卡颜色
    
    Args:
        color_value: 颜色值，可以是RGB[r,g,b]或LAB[L,a,b]
        method: 使用的颜色匹配方法 ('rgb', 'lab', 'cie2000')
        input_space: 输入颜色的颜色空间 ('rgb' or 'lab')
        
    Returns:
        tuple: (潘通颜色名称, RGB值, LAB值, 颜色距离)
    """
    min_distance = float('inf')
    closest_color = None
    closest_name = None
    closest_lab = None
    
    # 如果输入是RGB值，转换为LAB
    if input_space == 'rgb':
        lab1 = rgb_to_lab(color_value)
    else:
        lab1 = color_value
    
    for lab_color, pantone_name in PANTONE_COLORS:
        if method == 'rgb':
            rgb1 = lab_to_rgb(lab1)
            rgb2 = lab_to_rgb(lab_color)
            distance = np.sqrt(np.sum((rgb1 - rgb2) ** 2))
        elif method == 'lab':
            distance = np.sqrt(np.sum((np.array(lab1) - np.array(lab_color)) ** 2))
        elif method == 'cie2000':
            distance = delta_e_cie2000(lab1, lab_color)
        
        if distance < min_distance:
            min_distance = distance
            closest_lab = lab_color
            closest_color = lab_to_rgb(lab_color)
            closest_name = pantone_name
    
    return closest_name, closest_color, closest_lab, min_distance