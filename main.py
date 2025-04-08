"""
主程序文件 - 提供命令行接口来使用头发提取器
"""
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from hair_extractor import AnimeHairExtractor
from utils import load_image, display_image, save_image

os.environ['OMP_NUM_THREADS'] = '1'

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='从动漫人物图片中提取头发并进行颜色分析')
    parser.add_argument('image_path', type=str, help='输入图像的路径')
    parser.add_argument('--output', '-o', type=str, help='输出目录路径', default='output')
    parser.add_argument('--debug', '-d', action='store_true', help='显示调试信息和中间结果')
    parser.add_argument('--colors', '-c', type=int, default=3, help='提取的主要颜色数量')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.image_path):
        print(f"错误: 输入文件 '{args.image_path}' 不存在")
        return 1
    
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    print(f"正在处理图像: {args.image_path}")
    
    # 加载图像
    img_cv, img_pil = load_image(args.image_path)
    
    # 显示原始图像
    if args.debug:
        display_image(img_cv, '原始图像')
    
    # 创建头发提取器
    extractor = AnimeHairExtractor()
    
    # 提取头发
    print("正在提取头发区域...")
    hair_mask, hair_image = extractor.extract_hair(img_cv, debug=args.debug)
    
    # 提取主要颜色
    print("正在分析头发颜色...")
    dominant_colors = extractor.extract_dominant_color(hair_image, hair_mask, k=args.colors)
    
    if not dominant_colors:
        print("警告: 未能找到足够的头发区域来提取颜色")
    else:
        print(f"找到 {len(dominant_colors)} 种主要颜色:")
        for i, color_info in enumerate(dominant_colors):
            rgb = color_info['rgb']
            percentage = color_info.get('percentage', '未知')
            pantone_name = color_info.get('pantone_name', '')
            print(f"颜色 {i+1}: RGB = {rgb}, 占比 = {percentage}%, Pantone = {pantone_name}")
        
        # 主要颜色(占比最大的)
        main_color = dominant_colors[0]['rgb']
        main_pantone = dominant_colors[0].get('pantone_name', '')
        print(f"主要头发颜色: RGB = {main_color}, Pantone = {main_pantone}")
        
        # 可视化颜色
        extractor.visualize_colors(dominant_colors)
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # 保存头发掩码
    mask_path = os.path.join(args.output, f"{base_name}_hair_mask.png")
    cv2.imwrite(mask_path, hair_mask)
    print(f"头发掩码已保存至: {mask_path}")
    
    # 保存提取的头发
    hair_path = os.path.join(args.output, f"{base_name}_hair_extracted.png")
    cv2.imwrite(hair_path, hair_image)
    print(f"头发图像已保存至: {hair_path}")
    
    # 保存颜色信息
    if dominant_colors:
        # 创建颜色条可视化
        n_colors = len(dominant_colors)
        color_bar = np.zeros((100, n_colors * 100, 3), dtype=np.uint8)
        
        for i, color_info in enumerate(dominant_colors):
            rgb = np.array(color_info['rgb'])
            color_bar[:, i*100:(i+1)*100] = rgb
        
        color_path = os.path.join(args.output, f"{base_name}_hair_colors.png")
        cv2.imwrite(color_path, cv2.cvtColor(color_bar, cv2.COLOR_RGB2BGR))
        print(f"颜色信息已保存至: {color_path}")
        
        # 保存颜色信息到文本文件
        color_txt_path = os.path.join(args.output, f"{base_name}_hair_colors.txt")
        with open(color_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"图像: {args.image_path}\n")
            f.write(f"主要头发颜色 (RGB): {main_color}\n")
            if main_pantone:
                f.write(f"主要头发颜色 (Pantone): {main_pantone}\n")
            f.write("\n所有提取的颜色 (按占比排序):\n")
            for i, color_info in enumerate(dominant_colors):
                rgb = color_info['rgb']
                percentage = color_info.get('percentage', '未知')
                pantone_name = color_info.get('pantone_name', '')
                f.write(f"颜色 {i+1}: RGB = {rgb}, 占比 = {percentage}%")
                if pantone_name:
                    f.write(f", Pantone = {pantone_name}")
                f.write("\n")
        
        print(f"颜色文本信息已保存至: {color_txt_path}")
    
    print("处理完成!")
    return 0

if __name__ == "__main__":
    exit(main())