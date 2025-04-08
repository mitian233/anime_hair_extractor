"""
潘通颜色数据模块 - 提供潘通色卡颜色信息和匹配功能
"""
import numpy as np
from skimage import color
import math

# 潘通色卡颜色数据 [RGB值, 颜色名称]
PANTONE_COLORS = [
    # 红色系列
    [[237, 28, 36], "潘通 Red 032 C"],
    [[237, 41, 57], "潘通 Warm Red C"],
    [[218, 37, 29], "潘通 485 C"],
    [[193, 39, 45], "潘通 1788 C"],
    [[186, 12, 47], "潘通 200 C"],
    [[166, 9, 61], "潘通 207 C"],
    [[220, 88, 42], "潘通 179 C"],
    [[250, 70, 22], "潘通 Orange 021 C"],
    [[244, 115, 33], "潘通 1655 C"],
    
    # 橙色系列
    [[245, 130, 32], "潘通 165 C"],
    [[249, 157, 28], "潘通 1495 C"],
    [[249, 176, 0], "潘通 137 C"],
    
    # 黄色系列
    [[254, 221, 0], "潘通 Yellow C"],
    [[250, 224, 83], "潘通 100 C"],
    [[240, 228, 66], "潘通 102 C"],
    
    # 绿色系列
    [[196, 214, 0], "潘通 382 C"],
    [[141, 198, 63], "潘通 376 C"],
    [[0, 166, 81], "潘通 Green C"],
    [[0, 168, 133], "潘通 3278 C"],
    [[0, 133, 120], "潘通 328 C"],
    [[0, 115, 152], "潘通 3145 C"],
    
    # 蓝色系列
    [[0, 101, 179], "潘通 3015 C"],
    [[0, 113, 187], "潘通 Process Blue C"],
    [[0, 133, 202], "潘通 2925 C"],
    [[35, 31, 32], "潘通 Reflex Blue C"],
    [[65, 64, 153], "潘通 2728 C"],
    [[46, 49, 145], "潘通 286 C"],
    
    # 紫色系列
    [[146, 39, 143], "潘通 Purple C"],
    [[177, 7, 135], "潘通 241 C"],
    [[224, 0, 105], "潘通 Rubine Red C"],
    [[228, 0, 120], "潘通 Rhodamine Red C"],
    
    # 粉色系列
    [[230, 0, 126], "潘通 219 C"],
    [[235, 156, 197], "潘通 210 C"],
    [[244, 195, 0], "潘通 Yellow 012 C"],
    
    # 褐色系列
    [[128, 100, 30], "潘通 871 C"],
    [[143, 86, 64], "潘通 4705 C"],
    [[83, 40, 43], "潘通 490 C"],
    
    # 黑白灰系列
    [[0, 0, 0], "潘通 Black C"],
    [[255, 255, 255], "潘通 White"],
    [[35, 31, 32], "潘通 Black 6 C"],
    [[128, 130, 133], "潘通 Cool Gray 7 C"],
    [[167, 168, 170], "潘通 Cool Gray 4 C"],
    [[237, 237, 237], "潘通 Cool Gray 1 C"],
    
    # 动漫常见的特殊发色
    [[123, 104, 238], "潘通 薰衣草蓝"],
    [[255, 0, 255], "潘通 品红"],
    [[138, 43, 226], "潘通 紫罗兰"],
    [[64, 224, 208], "潘通 青绿"],
    [[255, 215, 0], "潘通 金色"],
    [[192, 192, 192], "潘通 银色"],
    [[112, 128, 144], "潘通 石板灰"],
    [[188, 143, 143], "潘通 浅玫瑰棕"],
    [[210, 180, 140], "潘通 棕褐色"],
    [[245, 222, 179], "潘通 小麦色"],
    [[255, 228, 225], "潘通 雾玫瑰"],
    [[255, 192, 203], "潘通 粉红"],
    [[240, 128, 128], "潘通 淡珊瑚色"],
    [[250, 128, 114], "潘通 鲑鱼色"],
    [[255, 20, 147], "潘通 深粉色"],
    [[199, 21, 133], "潘通 适中的紫红色"],
    [[153, 50, 204], "潘通 深兰花紫"],
    [[148, 0, 211], "潘通 深紫色"],
    [[106, 90, 205], "潘通 石板蓝"],
    [[72, 61, 139], "潘通 深岩蓝"],
    [[25, 25, 112], "潘通 午夜蓝"],
    [[0, 0, 128], "潘通 海军蓝"],
    [[0, 139, 139], "潘通 深青色"],
    [[107, 142, 35], "潘通 橄榄土褐色"],
    [[154, 205, 50], "潘通 黄绿色"],
    [[85, 107, 47], "潘通 深橄榄绿"],
    [[128, 128, 0], "潘通 橄榄色"],
    [[160, 82, 45], "潘通 赭色"],
    [[165, 42, 42], "潘通 棕色"],
    [[139, 69, 19], "潘通 马鞍棕色"],
    [[128, 0, 0], "潘通 栗色"],
]

def rgb_to_lab(rgb_color):
    """
    将RGB颜色转换为CIE Lab颜色空间
    
    Args:
        rgb_color: RGB颜色值 [r, g, b] (0-255)
        
    Returns:
        ndarray: Lab颜色值
    """
    # 归一化RGB值到0-1范围
    rgb_normalized = np.array(rgb_color) / 255.0
    # 转换为Lab空间
    lab_color = color.rgb2lab([[[rgb_normalized]]])
    return lab_color[0, 0]

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

def find_closest_pantone(rgb_color, method='lab'):
    """
    找出最接近给定RGB颜色的潘通色卡颜色
    
    Args:
        rgb_color: RGB颜色值 [r, g, b] (0-255)
        method: 使用的颜色匹配方法 ('rgb', 'lab', 'cie2000')
        
    Returns:
        tuple: (潘通颜色名称, RGB值, 颜色距离)
    """
    min_distance = float('inf')
    closest_color = None
    closest_name = None
    
    for pantone_color, pantone_name in PANTONE_COLORS:
        if method == 'rgb':
            distance = color_distance_rgb(rgb_color, pantone_color)
        elif method == 'lab':
            distance = color_distance_lab(rgb_color, pantone_color)
        elif method == 'cie2000':
            lab1 = rgb_to_lab(rgb_color)
            lab2 = rgb_to_lab(pantone_color)
            distance = delta_e_cie2000(lab1, lab2)
        
        if distance < min_distance:
            min_distance = distance
            closest_color = pantone_color
            closest_name = pantone_name
    
    return closest_name, closest_color, min_distance