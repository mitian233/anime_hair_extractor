"""
图像处理实用工具模块 - 提供基本的图像处理功能
"""
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import color, segmentation, measure, morphology
import matplotlib as mpl

# 设置 matplotlib 使用中文字体
def setup_chinese_font():
    """设置 matplotlib 使用中文字体"""
    # 尝试设置中文字体，按照常见中文字体的顺序尝试
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti', 'Source Han Sans CN']
    
    font_set = False
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.family'] = [font_name]
            # 验证字体是否可用
            fig = plt.figure()
            plt.text(0.5, 0.5, '测试中文', fontsize=12)
            plt.close(fig)
            font_set = True
            break
        except:
            continue
    
    if not font_set:
        print("警告：未能找到合适的中文字体，文字显示可能会有问题")
    
    # 修复负号显示
    mpl.rcParams['axes.unicode_minus'] = False

# 初始化中文字体设置
setup_chinese_font()

def load_image(image_path):
    """
    加载并返回图像，对于透明图片会添加白色底色
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        tuple: (BGR格式的OpenCV图像, RGB格式的PIL图像)
    """
    # 使用PIL读取图像（保留可能的Alpha通道）
    img_pil_original = Image.open(image_path)
    
    # 检查是否有Alpha通道（透明通道）
    if img_pil_original.mode == 'RGBA':
        # 创建一个白色背景
        white_bg = Image.new('RGBA', img_pil_original.size, (255, 255, 255, 255))
        # 将原图合成到白色背景上
        img_pil = Image.alpha_composite(white_bg, img_pil_original).convert('RGB')
    elif img_pil_original.mode == 'LA' or img_pil_original.mode == 'P':
        # 对于灰度+透明或者调色板模式，转换为RGBA再处理
        img_pil_rgba = img_pil_original.convert('RGBA')
        white_bg = Image.new('RGBA', img_pil_rgba.size, (255, 255, 255, 255))
        img_pil = Image.alpha_composite(white_bg, img_pil_rgba).convert('RGB')
    else:
        # 非透明图片直接转换为RGB
        img_pil = img_pil_original.convert('RGB')
    
    # 将PIL图像转换为OpenCV格式（BGR）
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_cv, img_pil

def display_image(image, title='Image', figsize=(10, 8)):
    """
    显示图像
    
    Args:
        image: numpy数组或PIL图像
        title: 图像标题
        figsize: 图像尺寸
    """
    plt.figure(figsize=figsize)
    
    # 如果是numpy数组，转换BGR到RGB（如果需要）
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 检查是否是BGR格式
            if np.max(image) > 1.0:  # 假设是0-255范围
                # 转换BGR到RGB
                image_to_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 已经是RGB或者其他三通道格式
                image_to_show = image
        else:
            image_to_show = image
    else:
        # PIL图像
        image_to_show = image
    
    plt.imshow(image_to_show)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image, save_path):
    """
    保存图像
    
    Args:
        image: 要保存的图像
        save_path: 保存路径
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 检查是否是BGR格式
            if np.max(image) > 1.0:  # 假设是0-255范围
                cv2.imwrite(save_path, image)
            else:
                # 缩放到0-255范围
                scaled = (image * 255).astype(np.uint8)
                cv2.imwrite(save_path, scaled)
        else:
            cv2.imwrite(save_path, image)
    else:
        # PIL图像
        image.save(save_path)