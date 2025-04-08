"""
头发提取模块 - 提供从动漫人物图片中提取头发的功能
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import segmentation, morphology, color, feature
import matplotlib.pyplot as plt
import mediapipe as mp
from pantone_colors import find_closest_pantone, rgb_to_lab

class AnimeHairExtractor:
    """
    动漫人物头发提取器 - 采用逐层分析方法
    """

    def __init__(self):
        """初始化头发提取器"""
        # 初始化MediaPipe人脸检测器
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_hair(self, image, debug=False):
        """
        从动漫人物图片中提取头发区域，使用逐层分析方法

        Args:
            image: BGR格式的OpenCV图像
            debug: 是否显示调试信息和中间结果

        Returns:
            tuple: (头发区域掩码, 分割后的头发图像)
        """
        # 步骤1: 预处理图像
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]

        # 步骤2: 提取人物轮廓（去除背景）
        character_mask = self._extract_character(image, debug)

        # 步骤3: 定位头部区域
        head_mask = self._locate_head_region(image, character_mask, debug)

        # 步骤4: 从头部区域中提取头发（去除面部和五官）
        hair_mask = self._extract_hair_from_head(image, head_mask, debug)

        # 应用形态学操作清理掩码
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        # 提取头发区域
        hair_extraction = cv2.bitwise_and(image, image, mask=final_mask)

        if debug:
            plt.figure(figsize=(15, 10))
            plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('原图', fontsize=10)
            plt.subplot(232), plt.imshow(character_mask, cmap='gray'), plt.title('人物轮廓', fontsize=10)
            plt.subplot(233), plt.imshow(head_mask, cmap='gray'), plt.title('头部区域', fontsize=10)
            plt.subplot(234), plt.imshow(hair_mask, cmap='gray'), plt.title('头发区域', fontsize=10)
            plt.subplot(235), plt.imshow(final_mask, cmap='gray'), plt.title('最终头发掩码', fontsize=10)
            plt.subplot(236), plt.imshow(cv2.cvtColor(hair_extraction, cv2.COLOR_BGR2RGB)), plt.title('提取的头发', fontsize=10)
            plt.tight_layout()
            plt.show()

        return final_mask, hair_extraction

    def _extract_character(self, image, debug=False):
        """
        从图像中提取人物轮廓，去除背景

        Args:
            image: BGR格式的OpenCV图像
            debug: 是否显示调试信息

        Returns:
            numpy.ndarray: 人物轮廓掩码
        """
        # 使用GrabCut自动分割前景(人物)和背景
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 初始化矩形 - 假设人物大致位于图像中央
        height, width = image.shape[:2]
        margin_x = width // 6
        margin_y = height // 6
        rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)

        # 执行GrabCut
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            # 如果GrabCut失败，使用简单的颜色阈值
            print("GrabCut分割失败，使用颜色阈值方法")
            return self._extract_character_by_color(image, debug)

        # 创建人物掩码 (前景和可能的前景)
        character_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        # 使用漫水填充法改进分割结果
        improved_mask = self._improve_segmentation(image, character_mask)

        if debug:
            plt.figure(figsize=(15, 5))
            plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('原图', fontsize=10)
            plt.subplot(132), plt.imshow(character_mask, cmap='gray'), plt.title('GrabCut分割', fontsize=10)
            plt.subplot(133), plt.imshow(improved_mask, cmap='gray'), plt.title('改进的分割', fontsize=10)
            plt.tight_layout()
            plt.show()

        return improved_mask

    def _extract_character_by_color(self, image, debug=False):
        """
        使用颜色阈值方法从图像中提取人物

        Args:
            image: BGR格式的OpenCV图像
            debug: 是否显示调试信息

        Returns:
            numpy.ndarray: 人物轮廓掩码
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建色彩直方图
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])

        # 找出主要的色调和饱和度范围
        h_peaks = self._find_histogram_peaks(hist_h, threshold=0.7)
        s_peaks = self._find_histogram_peaks(hist_s, threshold=0.7)

        # 创建各种颜色范围的掩码，捕获可能的人物区域
        masks = []
        for h_peak in h_peaks:
            h_low = max(0, h_peak - 10)
            h_high = min(180, h_peak + 10)
            for s_peak in s_peaks:
                s_low = max(0, s_peak - 30)
                s_high = min(255, s_peak + 30)

                # 饱和度较高的区域更可能是人物而非背景
                if s_peak > 30:
                    mask = cv2.inRange(hsv, (h_low, s_low, 50), (h_high, s_high, 255))
                    if np.sum(mask) > 1000:  # 排除太小的区域
                        masks.append(mask)

        if not masks:
            # 如果没有找到合适的掩码，使用简单的阈值分割
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask

        # 合并所有掩码
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # 形态学操作改进掩码
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # 保留最大的连通区域
        combined_mask = self._keep_largest_contour(combined_mask)

        return combined_mask

    def _find_histogram_peaks(self, histogram, threshold=0.7):
        """
        从直方图中找出峰值

        Args:
            histogram: 直方图数组
            threshold: 相对于最大值的阈值

        Returns:
            list: 峰值索引列表
        """
        # 平滑直方图
        smooth_hist = cv2.GaussianBlur(histogram, (5, 5), 0)
        # 找出峰值
        max_val = np.max(smooth_hist)
        threshold_val = max_val * threshold
        peaks = []
        for i in range(1, len(smooth_hist) - 1):
            if (smooth_hist[i] > smooth_hist[i-1] and
                smooth_hist[i] > smooth_hist[i+1] and
                smooth_hist[i] > threshold_val):
                peaks.append(i)
        return peaks

    def _improve_segmentation(self, image, initial_mask):
        """
        使用漫水填充法改进分割结果

        Args:
            image: BGR格式的OpenCV图像
            initial_mask: 初始分割掩码

        Returns:
            numpy.ndarray: 改进的分割掩码
        """
        # 使用初始掩码作为种子
        contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return initial_mask

        # 找出最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 创建新掩码
        improved_mask = np.zeros_like(initial_mask)
        cv2.drawContours(improved_mask, [largest_contour], 0, 255, -1)

        # 填充孔洞
        kernel = np.ones((5, 5), np.uint8)
        improved_mask = cv2.morphologyEx(improved_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return improved_mask

    def _locate_head_region(self, image, character_mask, debug=False):
        """
        从人物轮廓中定位头部区域

        Args:
            image: BGR格式的OpenCV图像
            character_mask: 人物轮廓掩码
            debug: 是否显示调试信息

        Returns:
            numpy.ndarray: 头部区域掩码
        """
        # 尝试使用MediaPipe进行人脸检测
        face_detected = False
        head_mask = np.zeros_like(character_mask)

        # 首先尝试使用MediaPipe人脸检测
        with self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)

            if results.detections:
                face_detected = True
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
                    width, height = int(bboxC.width * w), int(bboxC.height * h)

                    # 扩大检测区域，包含头发
                    x_expanded = max(0, x - width//2)
                    y_expanded = max(0, y - height)
                    width_expanded = min(w - x_expanded, width * 2)
                    height_expanded = min(h - y_expanded, height * 2)

                    # 创建头部掩码
                    head_mask[y_expanded:y_expanded+height_expanded, x_expanded:x_expanded+width_expanded] = 255

        # 如果MediaPipe检测失败，使用基于形态的方法
        if not face_detected:
            # 查找人物轮廓
            contours, _ = cv2.findContours(character_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return character_mask  # 如果没有找到轮廓，返回整个人物掩码

            # 找出最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)

            # 找出人物的上部(假设为头部)
            x, y, w, h = cv2.boundingRect(largest_contour)
            head_height = h // 3  # 假设头部占人物高度的1/3
            head_mask = np.zeros_like(character_mask)
            head_mask[y:y+head_height, x:x+w] = 255

            # 与人物掩码相交，确保只包含人物区域
            head_mask = cv2.bitwise_and(head_mask, character_mask)

        if debug:
            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(character_mask, cmap='gray'), plt.title('人物掩码', fontsize=10)
            plt.subplot(122), plt.imshow(head_mask, cmap='gray'), plt.title('头部区域', fontsize=10)
            plt.tight_layout()
            plt.show()

        return head_mask

    def _extract_hair_from_head(self, image, head_mask, debug=False):
        """
        从头部区域中提取头发，去除面部和五官

        Args:
            image: BGR格式的OpenCV图像
            head_mask: 头部区域掩码
            debug: 是否显示调试信息

        Returns:
            numpy.ndarray: 头发区域掩码
        """
        # 使用颜色信息从头部区域提取头发
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建候选头发区域掩码
        # 黑色/深色头发
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask_black = cv2.inRange(img_hsv, lower_black, upper_black)

        # 金色/黄色头发
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        # 棕色头发
        lower_brown = np.array([5, 50, 50])
        upper_brown = np.array([20, 255, 200])
        mask_brown = cv2.inRange(img_hsv, lower_brown, upper_brown)

        # 蓝色头发
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

        # 粉色/紫色头发
        lower_pink = np.array([140, 50, 50])
        upper_pink = np.array([170, 255, 255])
        mask_pink = cv2.inRange(img_hsv, lower_pink, upper_pink)

        # 红色头发 (需要两个范围因为红色在HSV环绕)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 绿色头发
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

        # 白色/灰色头发
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)

        # 合并所有颜色掩码
        color_masks = [mask_black, mask_yellow, mask_brown, mask_blue,
                       mask_pink, mask_red, mask_green, mask_white]
        hair_color_mask = np.zeros_like(mask_black)
        for mask in color_masks:
            hair_color_mask = cv2.bitwise_or(hair_color_mask, mask)

        # 与头部掩码相交，确保只考虑头部区域
        hair_mask = cv2.bitwise_and(hair_color_mask, head_mask)

        # 尝试检测面部区域，去除脸部
        face_mask = self._detect_face(image, debug)
        if np.sum(face_mask) > 0:  # 如果检测到脸部
            # 从头发掩码中去除脸部
            hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(face_mask))

        # 形态学操作改进头发掩码
        kernel = np.ones((3, 3), np.uint8)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 保留最大的连通区域和较大的区域
        hair_mask = self._keep_significant_regions(hair_mask)

        if debug:
            plt.figure(figsize=(15, 5))
            plt.subplot(141), plt.imshow(head_mask, cmap='gray'), plt.title('头部区域', fontsize=10)
            plt.subplot(142), plt.imshow(hair_color_mask, cmap='gray'), plt.title('颜色掩码', fontsize=10)
            plt.subplot(143), plt.imshow(face_mask, cmap='gray'), plt.title('面部区域', fontsize=10)
            plt.subplot(144), plt.imshow(hair_mask, cmap='gray'), plt.title('头发掩码', fontsize=10)
            plt.tight_layout()
            plt.show()

        return hair_mask

    def _detect_face(self, image, debug=False):
        """
        检测面部区域

        Args:
            image: BGR格式的OpenCV图像
            debug: 是否显示调试信息

        Returns:
            numpy.ndarray: 面部区域掩码
        """
        face_mask = np.zeros(image.shape[:2], np.uint8)

        # 尝试使用MediaPipe人脸网格检测
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w = image.shape[:2]
                    face_points = []
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        face_points.append((x, y))

                    # 将点转换为numpy数组
                    face_points = np.array(face_points)

                    # 找出面部轮廓
                    hull = cv2.convexHull(face_points)
                    cv2.drawContours(face_mask, [hull], 0, 255, -1)
            else:
                # 如果MediaPipe检测失败，尝试使用Haar级联分类器
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    # 稍微扩大人脸区域
                    x_expanded = max(0, x - w//10)
                    y_expanded = max(0, y - h//10)
                    w_expanded = min(image.shape[1] - x_expanded, w * 1.2)
                    h_expanded = min(image.shape[0] - y_expanded, h * 1.2)

                    cv2.rectangle(face_mask, (x_expanded, y_expanded),
                                 (x_expanded + w_expanded, y_expanded + h_expanded), 255, -1)

        # 应用形态学操作改进掩码
        kernel = np.ones((5, 5), np.uint8)
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)

        return face_mask

    def _keep_largest_contour(self, mask):
        """
        只保留掩码中最大的轮廓

        Args:
            mask: 二值掩码

        Returns:
            numpy.ndarray: 只包含最大轮廓的掩码
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        # 找出最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 创建新掩码
        result = np.zeros_like(mask)
        cv2.drawContours(result, [largest_contour], 0, 255, -1)

        return result

    def _keep_significant_regions(self, mask, min_area_ratio=0.05):
        """
        保留掩码中足够大的区域

        Args:
            mask: 二值掩码
            min_area_ratio: 相对于最大区域的最小面积比例

        Returns:
            numpy.ndarray: 只包含显著区域的掩码
        """
        # 寻找所有轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        # 计算最大区域面积
        max_area = max([cv2.contourArea(c) for c in contours])
        min_area = max_area * min_area_ratio

        # 创建新掩码，只包含足够大的区域
        result = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(result, [contour], 0, 255, -1)

        return result
    
    def extract_dominant_color(self, hair_image, mask, k=3, match_pantone=True, color_match_method='cie2000'):
        """
        从提取的头发图像中提取主要颜色，并匹配潘通色卡
        
        Args:
            hair_image: BGR格式的头发图像
            mask: 头发区域掩码
            k: 要提取的颜色数量
            match_pantone: 是否匹配潘通色卡颜色
            color_match_method: 颜色匹配方法 ('rgb', 'lab', 'cie2000')
            
        Returns:
            list: 主要颜色列表，每个颜色是一个字典，包含RGB值、LAB值、占比和潘通匹配结果
        """
        # 转换为RGB和LAB
        rgb_image = cv2.cvtColor(hair_image, cv2.COLOR_BGR2RGB)
        
        # 只选择掩码区域的像素
        pixels = rgb_image[mask == 255].reshape(-1, 3)
        
        # 如果没有足够的像素，返回空
        if len(pixels) < k:
            return []
        
        # 使用K-means聚类提取主要颜色
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # 获取聚类中心（颜色）
        colors = kmeans.cluster_centers_.astype(int)
        
        # 计算每个颜色的像素数量
        labels = kmeans.labels_
        counts = np.bincount(labels)
        total_pixels = np.sum(counts)
        
        # 按照像素计数排序颜色
        colors_with_counts = [(colors[i], counts[i], counts[i]/total_pixels*100) for i in range(len(colors))]
        colors_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        # 提取排序后的颜色并添加潘通匹配结果
        dominant_colors = []
        for color, count, percentage in colors_with_counts:
            # 将RGB转换为LAB
            lab_color = rgb_to_lab(color.tolist())  # 使用从pantone_colors导入的函数
            color_info = {
                'rgb': color.tolist(),
                'lab': lab_color.tolist(),
                'count': int(count),
                'percentage': float(percentage)
            }
            
            # 匹配潘通色卡
            if match_pantone:
                pantone_name, pantone_rgb, pantone_lab, distance = find_closest_pantone(
                    color.tolist(), 
                    method=color_match_method
                )
                color_info['pantone_name'] = pantone_name
                color_info['pantone_rgb'] = pantone_rgb.tolist()
                color_info['pantone_lab'] = pantone_lab
                color_info['color_distance'] = float(distance)
            
            dominant_colors.append(color_info)
        
        return dominant_colors

    def visualize_colors(self, colors, figsize=(15, 3)):
        """
        可视化颜色列表，并显示潘通色卡匹配结果
        
        Args:
            colors: 颜色列表，每个颜色是一个字典，包含RGB值、LAB值和潘通匹配结果
            figsize: 图像大小
        """
        if not colors:
            print("没有找到颜色")
            return
        
        # 创建色块图像
        n_colors = len(colors)
        plt.figure(figsize=figsize)
        
        for i, color_info in enumerate(colors):
            plt.subplot(1, n_colors, i+1)
            
            rgb = color_info['rgb']
            lab = color_info['lab']
            
            # 显示原始颜色
            bar_height = 0.6
            plt.bar(0, bar_height, color=[c/255 for c in rgb], width=0.8)
            
            # 如果有潘通色卡匹配，显示匹配的潘通颜色
            if 'pantone_rgb' in color_info:
                pantone_rgb = color_info['pantone_rgb']
                pantone_lab = color_info['pantone_lab']
                plt.bar(0, -bar_height, color=[c/255 for c in pantone_rgb], width=0.8, bottom=bar_height)
                
                # 添加颜色信息
                title = f'RGB: {rgb}\nLAB: [{lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}]\n{color_info["pantone_name"]}'
                if 'percentage' in color_info:
                    title = f'{color_info["percentage"]:.1f}%\n{title}'
            else:
                title = f'RGB: {rgb}\nLAB: [{lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}]'
                if 'percentage' in color_info:
                    title = f'{color_info["percentage"]:.1f}%\n{title}'
            
            plt.title(title, fontsize=10, pad=10)  # 调整字体大小和标题间距
            plt.xticks([])
            plt.yticks([])
        
        plt.tight_layout()
        plt.show()
    
    def match_pantone_colors(self, colors, method='cie2000'):
        """
        为提取的颜色匹配潘通色卡
        
        Args:
            colors: 颜色列表，每个元素可以是RGB数组或者包含'rgb'键的字典
            method: 匹配方法 ('rgb', 'lab', 'cie2000')
            
        Returns:
            list: 匹配结果列表，每个元素是一个字典，包含原始颜色和匹配的潘通色卡信息
        """
        matched_colors = []
        
        for color_item in colors:
            # 判断输入是RGB数组还是字典
            if isinstance(color_item, dict) and 'rgb' in color_item:
                rgb_color = color_item['rgb']
                result = dict(color_item)  # 复制原始字典
            else:
                rgb_color = color_item
                result = {'rgb': rgb_color}
            
            # 匹配潘通色卡
            pantone_name, pantone_rgb, distance = find_closest_pantone(rgb_color, method=method)
            
            result['pantone_name'] = pantone_name
            result['pantone_rgb'] = pantone_rgb
            result['color_distance'] = float(distance)
            
            matched_colors.append(result)
        
        return matched_colors