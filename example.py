"""
示例脚本 - 提供一个简单的例子来演示如何使用头发提取器
"""
import os
import sys
import matplotlib.pyplot as plt
from anime_hair_extractor.hair_extractor import AnimeHairExtractor
from anime_hair_extractor.utils import load_image, display_image

def run_example(image_path):
    """运行头发提取示例"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片 '{image_path}' 不存在")
        return
    
    print(f"正在处理图像: {image_path}")
    
    # 加载图像
    img_cv, img_pil = load_image(image_path)
    
    # 显示原始图像
    print("显示原始图像...")
    display_image(img_cv, '原始图像')
    
    # 创建头发提取器
    extractor = AnimeHairExtractor()
    
    # 提取头发
    print("正在提取头发区域...")
    hair_mask, hair_image = extractor.extract_hair(img_cv, debug=True)
    
    # 提取主要颜色
    print("正在分析头发颜色...")
    dominant_colors = extractor.extract_dominant_color(hair_image, hair_mask, k=5)
    
    if not dominant_colors:
        print("警告: 未能找到足够的头发区域来提取颜色")
    else:
        print(f"找到 {len(dominant_colors)} 种主要颜色:")
        for i, color in enumerate(dominant_colors):
            print(f"颜色 {i+1}: RGB = {color}")
        
        # 主要颜色(占比最大的)
        main_color = dominant_colors[0]
        print(f"主要头发颜色: RGB = {main_color}")
        
        # 可视化颜色
        print("显示提取的主要颜色...")
        extractor.visualize_colors(dominant_colors)
    
    print("示例运行完成!")

if __name__ == "__main__":
    # 如果提供了命令行参数，使用它作为图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 默认使用示例图像
        print("未提供图像路径，请提供一个动漫人物图片路径作为参数")
        print("用法: python example.py <图片路径>")
        sys.exit(1)
    
    run_example(image_path)