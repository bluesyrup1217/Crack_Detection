from detector import ImageBrightnessDetector

image_path = 'material/n120normal.jpg'
detector = ImageBrightnessDetector(image_path, 50)  # 128 fig
print(detector.get_brightness_status())
