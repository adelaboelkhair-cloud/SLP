"""
Main Application - PyQt5 GUI for Sign Language Production
التطبيق الرئيسي - واجهة المستخدم
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QFileDialog, QStatusBar,
    QProgressBar, QTabWidget, QTextEdit
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from pathlib import Path

from src.gesture_recognizer import GestureRecognizer
from config import APP_CONFIG, MEDIAPIPE_CONFIG, GESTURES


class VideoProcessorThread(QThread):
    """خيط معالجة الفيديو"""
    
    frame_signal = pyqtSignal(np.ndarray)
    gesture_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    
    def __init__(self, recognizer, video_source=0):
        super().__init__()
        self.recognizer = recognizer
        self.video_source = video_source
        self.is_running = True
    
    def run(self):
        """تشغيل معالجة الفيديو"""
        for results in self.recognizer.recognize_from_video(self.video_source):
            if not self.is_running:
                break
            
            self.frame_signal.emit(results["frame"])
            self.gesture_signal.emit(results)
    
    def stop(self):
        """إيقاف المعالجة"""
        self.is_running = False


class SLPMainWindow(QMainWindow):
    """النافذة الرئيسية للتطبيق"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_CONFIG["window_title"])
        self.setGeometry(100, 100, APP_CONFIG["window_width"], APP_CONFIG["window_height"])
        
        # تهيئة المتغيرات
        self.recognizer = None
        self.video_thread = None
        self.current_frame = None
        self.is_running = False
        
        # إنشاء الواجهة
        self.init_ui()
        
        # تحميل النموذج
        self.load_model()
    
    def init_ui(self):
        """إنشاء واجهة المستخدم"""
        # إنشاء الويدجت المركزي
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # إنشاء التخطيط الرئيسي
        main_layout = QHBoxLayout()
        
        # القسم الأيسر - عرض الفيديو
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        left_layout.addWidget(QLabel("Video Stream"))
        left_layout.addWidget(self.video_label)
        
        # أزرار التحكم
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_video)
        button_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_video)
        self.pause_btn.setEnabled(False)
        button_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(button_layout)
        
        # القسم الأيمن - معلومات الإشارات والإعدادات
        right_layout = QVBoxLayout()
        
        # علامات التبويب
        self.tabs = QTabWidget()
        
        # علامة التبويب 1 - النتائج
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Arial", 10))
        results_layout.addWidget(QLabel("Recognized Gestures:"))
        results_layout.addWidget(self.results_text)
        
        results_tab.setLayout(results_layout)
        self.tabs.addTab(results_tab, "Results")
        
        # علامة التبويب 2 - الإعدادات
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        
        # اختيار الكاميرا
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Webcam (0)", "Camera 1", "Camera 2"])
        camera_layout.addWidget(self.camera_combo)
        settings_layout.addLayout(camera_layout)
        
        # حد الثقة
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_label = QLabel("0.50")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v/100:.2f}")
        )
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_label)
        settings_layout.addLayout(confidence_layout)
        
        # أزرار إضافية
        load_video_btn = QPushButton("Load Video File")
        load_video_btn.clicked.connect(self.load_video_file)
        settings_layout.addWidget(load_video_btn)
        
        save_output_btn = QPushButton("Save Output")
        save_output_btn.clicked.connect(self.save_output)
        settings_layout.addWidget(save_output_btn)
        
        settings_layout.addStretch()
        settings_tab.setLayout(settings_layout)
        self.tabs.addTab(settings_tab, "Settings")
        
        right_layout.addWidget(self.tabs)
        
        # شريط التقدم
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # إضافة الأقسام إلى التخطيط الرئيسي
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        
        central_widget.setLayout(main_layout)
        
        # شريط الحالة
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
    
    def load_model(self):
        """تحميل النموذج المدرب"""
        try:
            self.recognizer = GestureRecognizer()
            self.statusBar.showMessage("Model loaded successfully")
        except Exception as e:
            self.statusBar.showMessage(f"Error loading model: {str(e)}")
    
    def start_video(self):
        """بدء معالجة الفيديو"""
        if not self.recognizer:
            self.statusBar.showMessage("Error: Model not loaded")
            return
        
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # إنشاء خيط معالجة الفيديو
        self.video_thread = VideoProcessorThread(self.recognizer)
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.gesture_signal.connect(self.update_results)
        self.video_thread.start()
        
        self.statusBar.showMessage("Video processing started")
    
    def pause_video(self):
        """إيقاف مؤقت لمعالجة الفيديو"""
        self.is_running = False
        self.statusBar.showMessage("Video processing paused")
    
    def stop_video(self):
        """إيقاف معالجة الفيديو"""
        self.is_running = False
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        self.statusBar.showMessage("Video processing stopped")
    
    def update_frame(self, frame):
        """تحديث الإطار المعروض"""
        self.current_frame = frame
        
        # تحويل الإطار لعرضه في PyQt5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
    
    def update_results(self, results):
        """تحديث نتائج التعرف"""
        text = ""
        
        if results["gestures"]:
            text += f"<b>Detected {results['num_hands']} hand(s):</b><br>"
            for gesture_info in results["gestures"]:
                text += f"<b>{gesture_info['gesture']}</b> "
                text += f"({gesture_info['hand']}) - "
                text += f"Confidence: {gesture_info['confidence']:.2%}<br>"
        else:
            text += "No gestures detected"
        
        self.results_text.setHtml(text)
    
    def load_video_file(self):
        """تحميل ملف فيديو"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if file_path:
            self.statusBar.showMessage(f"Loaded: {Path(file_path).name}")
    
    def save_output(self):
        """حفظ الفيديو المعالج"""
        if not self.current_frame is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Output",
                "",
                "Video Files (*.mp4);;Image Files (*.jpg)"
            )
            
            if file_path:
                cv2.imwrite(file_path, self.current_frame)
                self.statusBar.showMessage(f"Saved to: {Path(file_path).name}")
    
    def closeEvent(self, event):
        """معالج إغلاق النافذة"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        
        if self.recognizer:
            self.recognizer.close()
        
        event.accept()


def main():
    """نقطة البداية للتطبيق"""
    app = QApplication(sys.argv)
    
    # تطبيق الأسلوب
    app.setStyle('Fusion')
    
    # إنشاء والنافذة الرئيسية
    window = SLPMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
