import sys
import cv2
import torch
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import numpy as np


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("图像识别工具")
        self.setGeometry(100, 100, 800, 600)

        # 模型加载（假设你用的是 YOLOv5）
        self.model = torch.hub.load(
            r'C:\Users\21435\Desktop\railway2\yolov5-master',  # 本地 yolov5 目录（含 hubconf.py）
            'custom',
            path=r'C:\Users\21435\Desktop\railway2\yolov5-master\runs\train\exp7\weights\best.pt',
            source='local'
        )# 替换成你自己的模型路径

        # UI 元素
        self.label = QLabel("请上传图片", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.clicked.connect(self.load_image)

        self.detect_btn = QPushButton("识别图像")
        self.detect_btn.clicked.connect(self.detect_image)
        self.detect_btn.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.detect_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.img_path = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.img_path = file_path
            pixmap = QPixmap(file_path).scaled(640, 480, Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)
            self.detect_btn.setEnabled(True)

    def detect_image(self):
        if not self.img_path:
            return

        # 用模型识别
        results = self.model(self.img_path)
        results.render()  # 在图像上绘制检测框

        # OpenCV 图像格式转 Qt 显示格式
        img = results.ims[0]  # numpy array (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(qt_image).scaled(640, 480, Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())