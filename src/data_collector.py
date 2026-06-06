"""
Data Collection Module - Collect hand gesture data for training
وحدة جمع البيانات - جمع بيانات الإشارات لتدريب النموذج
"""

import cv2
import numpy as np
from pathlib import Path
from src.mediapipe_processor import MediaPipeProcessor
from src.utils import create_directories, save_landmarks_to_file, draw_text_box
from config import DATA_COLLECTION_CONFIG, GESTURES


class DataCollector:
    """أداة لجمع بيانات الإشارات"""
    
    def __init__(self, output_dir="data/"):
        """
        تهيئة جامع البيانات
        
        Args:
            output_dir: مجلد حفظ البيانات
        """
        self.processor = MediaPipeProcessor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # إنشاء مجلدات للإشارات
        for gesture in GESTURES:
            (self.output_dir / gesture).mkdir(exist_ok=True)
    
    def collect_gesture_data(self, gesture_name, num_samples=None):
        """
        جمع بيانات لإشارة محددة
        
        Args:
            gesture_name: اسم الإشارة
            num_samples: عدد العينات (None = استخدام الإعداد الافتراضي)
        """
        if num_samples is None:
            num_samples = DATA_COLLECTION_CONFIG["samples_per_gesture"]
        
        cap = cv2.VideoCapture(0)
        collected_count = 0
        
        print(f"Starting data collection for gesture: {gesture_name}")
        print(f"Target: {num_samples} samples")
        
        while collected_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # معالجة الإطار
            results = self.processor.process_frame(frame)
            annotated_frame = results["annotated_frame"]
            
            # إضافة معلومات على الشاشة
            text = f"Gesture: {gesture_name} | Samples: {collected_count}/{num_samples}"
            annotated_frame = draw_text_box(annotated_frame, text, (10, 50))
            
            # عرض الإطار
            cv2.imshow("Data Collection", annotated_frame)
            
            # الضغط على 'c' لجمع العينة
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if results["detected"]:
                    for landmarks in results["landmarks"]:
                        # تطبيع النقاط
                        normalized = self.processor.normalize_landmarks(landmarks)
                        
                        # حفظ البيانات
                        filename = save_landmarks_to_file(
                            normalized,
                            gesture_name,
                            self.output_dir
                        )
                        print(f"Sample {collected_count + 1} saved: {filename}")
                        collected_count += 1
                else:
                    print("No hand detected! Try again.")
            
            # الضغط على 'q' للخروج
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Data collection completed. Total samples: {collected_count}")
    
    def collect_all_gestures(self):
        """جمع بيانات لجميع الإشارات"""
        for gesture in GESTURES:
            input(f"Press Enter to start collecting data for '{gesture}'...")
            self.collect_gesture_data(gesture)
    
    def get_statistics(self):
        """الحصول على إحصائيات البيانات المجمعة"""
        stats = {}
        
        for gesture in GESTURES:
            gesture_dir = self.output_dir / gesture
            if gesture_dir.exists():
                samples = list(gesture_dir.glob("*.npy"))
                stats[gesture] = len(samples)
        
        return stats
    
    def close(self):
        """إغلاق الموارد"""
        self.processor.close()


# مثال على الاستخدام
if __name__ == "__main__":
    collector = DataCollector()
    
    # عرض الإحصائيات الحالية
    print("Current data statistics:")
    stats = collector.get_statistics()
    for gesture, count in stats.items():
        print(f"  {gesture}: {count} samples")
    
    # جمع البيانات
    print("\\nStarting data collection...")
    collector.collect_all_gestures()
    
    # عرض الإحصائيات النهائية
    print("\\nFinal data statistics:")
    stats = collector.get_statistics()
    for gesture, count in stats.items():
        print(f"  {gesture}: {count} samples")
    
    collector.close()
