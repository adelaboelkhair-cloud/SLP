"""
Camera Handler - Manages video capture and frame processing
معالج الكاميرا - إدارة التقاط الفيديو ومعالجة الإطارات
"""

import cv2
import numpy as np
from pathlib import Path
from config import VIDEO_CONFIG


class CameraHandler:
    """معالج الكاميرا والفيديو"""
    
    def __init__(self, camera_index=0, width=None, height=None, fps=None):
        """
        تهيئة معالج الكاميرا
        
        Args:
            camera_index: رقم الكاميرا (0 = الكاميرا الافتراضية)
            width: عرض الإطار
            height: ارتفاع الإطار
            fps: معدل الإطارات
        """
        self.camera_index = camera_index
        self.width = width or VIDEO_CONFIG["frame_width"]
        self.height = height or VIDEO_CONFIG["frame_height"]
        self.fps = fps or VIDEO_CONFIG["fps"]
        
        self.cap = cv2.VideoCapture(camera_index)
        
        # تعيين خصائص الكاميرا
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # تحسين جودة الكاميرا
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
    
    def get_frame(self):
        """
        الحصول على إطار واحد من الكاميرا
        
        Returns:
            tuple: (success, frame)
        """
        ret, frame = self.cap.read()
        
        if ret:
            # قلب الإطار أفقياً
            if VIDEO_CONFIG["flip_frame"]:
                frame = cv2.flip(frame, 1)
        
        return ret, frame
    
    def get_frames_generator(self, max_frames=None):
        """
        الحصول على سلسلة من الإطارات
        
        Args:
            max_frames: أقصى عدد إطارات (None = بلا حد)
            
        Yields:
            np.array: الإطارات
        """
        frame_count = 0
        
        while True:
            ret, frame = self.get_frame()
            
            if not ret:
                break
            
            yield frame
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
    
    def load_video_file(self, file_path):
        """
        تحميل ملف فيديو
        
        Args:
            file_path: مسار ملف الفيديو
            
        Returns:
            bool: النجاح
        """
        self.cap.release()
        self.cap = cv2.VideoCapture(str(file_path))
        
        return self.cap.isOpened()
    
    def save_video(self, output_path, frames, codec='mp4v'):
        """
        حفظ الإطارات كملف فيديو
        
        Args:
            output_path: مسار الحفظ
            frames: قائمة الإطارات
            codec: كود الفيديو
        """
        if not frames:
            print("No frames to save")
            return
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Video saved to {output_path}")
    
    def get_camera_properties(self):
        """الحصول على خصائص الكاميرا"""
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
        }
    
    def set_brightness(self, value):
        """تعيين درجة السطوع"""
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, value)
    
    def set_contrast(self, value):
        """تعيين التباين"""
        self.cap.set(cv2.CAP_PROP_CONTRAST, value)
    
    def set_saturation(self, value):
        """تعيين التشبع"""
        self.cap.set(cv2.CAP_PROP_SATURATION, value)
    
    def record_frames(self, num_frames, output_path=None):
        """
        تسجيل عدد معين من الإطارات
        
        Args:
            num_frames: عدد الإطارات المراد تسجيلها
            output_path: مسار الحفظ (اختياري)
            
        Returns:
            list: قائمة الإطارات
        """
        frames = []
        
        for frame in self.get_frames_generator(num_frames):
            frames.append(frame)
        
        if output_path:
            self.save_video(output_path, frames)
        
        return frames
    
    def release(self):
        """تحرير موارد الكاميرا"""
        self.cap.release()
    
    def __del__(self):
        """تنظيف الموارد عند حذف الكائن"""
        self.release()


# مثال على الاستخدام
if __name__ == "__main__":
    # إنشاء معالج الكاميرا
    camera = CameraHandler()
    
    print("Camera properties:")
    props = camera.get_camera_properties()
    for key, value in props.items():
        print(f"  {key}: {value}")
    
    print("\\nCapturing frames from camera...")
    print("Press 'q' to exit")
    
    frame_count = 0
    
    for frame in camera.get_frames_generator():
        # عرض معلومات على الإطار
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # عرض الإطار
        cv2.imshow("Camera Feed", frame)
        
        # للخروج
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cv2.destroyAllWindows()
    camera.release()
    
    print(f"Total frames captured: {frame_count}")
