import cv2
import numpy as np


class ImageBrightnessDetector:
    # 初始化参数
    def __init__(self, image_path, mid_value):
        self.image_path = image_path
        self.img = None
        self.gray_img = None
        self.img_shape = None
        self.height = None
        self.width = None
        self.size = None
        self.hist = None
        self.a = 0
        self.ma = 0
        self.reduce_matrix = None
        self.shift_value = None
        self.shift_sum = None
        self.da = None
        self.m = None
        self.k = None
        self.gray_value = None
        self.mid_values = mid_value

    def calculate_brightness(self):
        # 拿图，转换成灰度图，赋予基本图像参数信息
        self.img = cv2.imread(self.image_path)
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gray_value = cv2.mean(self.gray_img)
        self.img_shape = self.gray_img.shape
        self.height, self.width = self.img_shape[0], self.img_shape[1]
        self.size = self.gray_img.size  # 灰度图的像素数
        # 灰度图的直方图
        self.hist = cv2.calcHist([self.gray_img], [0], None, [256], [0, 256])
        # 灰度图级别为256，暂时以中间值128划分明暗大概
        self.reduce_matrix = np.full((self.height, self.width), self.mid_values)  # 创建np数组，用128的值填充，中间灰度值128
        self.shift_value = self.gray_img - self.reduce_matrix  # 计算每个像素点灰度值相较于中间灰度值128的差值（可做微调）
        self.shift_sum = np.sum(self.shift_value)  # 所有差值的总和
        self.da = self.shift_sum / self.size  # 整个灰度图像的平均偏差值
        # 计算偏离128的平均偏差
        for i in range(256):
            self.ma += (abs(i - self.mid_values - self.da) * self.hist[i])  # 128为中间灰度值，*对应灰度级别的像素数
        self.m = abs(self.ma / self.size)  # 平均绝对偏差
        self.k = abs(self.da) / self.m  # k为亮度系数，abs保证k为正

    def get_brightness_status(self):
        self.calculate_brightness()
        print('该图片的整体灰度值为：', int(self.gray_value[0]))
        if self.k[0] > 1:
            if self.da > 0:  # 大于0说明整体图像亮度在128之上
                return 'normal-extra'
            else:
                return 'an'
        else:  # ＜1说明平均偏差小于绝对偏差
            return 'normal'
