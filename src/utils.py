"""
Utility functions for SLP
دوال مساعدة عامة
"""

import numpy as np
import cv2
from pathlib import Path


def create_directories(base_path="./"):
    """
    إنشاء المجلدات الأساسية للمشروع
    
    Args:
        base_path: المسار الأساسي
    """
    directories = [
        "data/train",
        "data/test",
        "data/validation",
        "models",
        "logs",
        "output"
    ]
    
    for directory in directories:
        Path(base_path) / directory
        (Path(base_path) / directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def save_landmarks_to_file(landmarks, gesture_name, output_dir="data/"):
    """
    حفظ النقاط إلى ملف
    
    Args:
        landmarks: النقاط المستخرجة
        gesture_name: اسم الإشارة
        output_dir: مجلد الحفظ
    """
    output_path = Path(output_dir) / gesture_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # حفظ النقاط كملف نصي
    filename = output_path / f"{gesture_name}_{len(list(output_path.glob('*.npy')))}.npy"
    np.save(filename, landmarks)
    
    return filename


def load_landmarks_from_file(file_path):
    """
    تحميل النقاط من ملف
    
    Args:
        file_path: مسار الملف
        
    Returns:
        np.array: النقاط المحملة
    """
    return np.load(file_path)


def resize_frame(frame, width=640, height=480):
    """
    تغيير حجم الإطار
    
    Args:
        frame: الإطار الأصلي
        width: العرض الجديد
        height: الارتفاع الجديد
        
    Returns:
        np.array: الإطار المعاد تحجيمه
    """
    return cv2.resize(frame, (width, height))


def flip_frame(frame, flip_code=1):
    """
    قلب الإطار أفقياً أو عمودياً
    
    Args:
        frame: الإطار الأصلي
        flip_code: 1 = أفقي، 0 = عمودي، -1 = كليهما
        
    Returns:
        np.array: الإطار المقلوب
    """
    return cv2.flip(frame, flip_code)


def calculate_distance(point1, point2):
    """
    حساب المسافة بين نقطتين
    
    Args:
        point1: النقطة الأولى (x, y)
        point2: النقطة الثانية (x, y)
        
    Returns:
        float: المسافة
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(point1, point2, point3):
    """
    حساب الزاوية بين ثلاث نقاط
    
    Args:
        point1: النقطة الأولى
        point2: النقطة الوسطى
        point3: النقطة الثالثة
        
    Returns:
        float: الزاوية بالدرجات
    """
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=5):
    """
    رسم النقاط على الإطار
    
    Args:
        frame: الإطار
        landmarks: النقاط المراد رسمها
        color: لون النقاط (BGR)
        radius: نصف قطر النقطة
        
    Returns:
        np.array: الإطار مع النقاط المرسومة
    """
    for landmark in landmarks:
        x = int(landmark[0] * frame.shape[1])
        y = int(landmark[1] * frame.shape[0])
        cv2.circle(frame, (x, y), radius, color, -1)
    
    return frame


def draw_text_box(frame, text, position=(10, 50), font_scale=1, thickness=2, 
                  bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """
    رسم نص مع خلفية على الإطار
    
    Args:
        frame: الإطار
        text: النص
        position: موضع النص
        font_scale: حجم الخط
        thickness: سمك الخط
        bg_color: لون الخلفية
        text_color: لون النص
        
    Returns:
        np.array: الإطار مع النص
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # حساب حجم النص
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # رسم الخلفية
    cv2.rectangle(
        frame,
        (position[0] - 5, position[1] - text_height - 5),
        (position[0] + text_width + 5, position[1] + baseline + 5),
        bg_color,
        -1
    )
    
    # رسم النص
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)
    
    return frame


def normalize_landmarks(landmarks, method='min_max'):
    """
    تطبيع النقاط
    
    Args:
        landmarks: النقاط الخام
        method: طريقة التطبيع ('min_max' أو 'standard')
        
    Returns:
        np.array: النقاط المطبعة
    """
    if method == 'min_max':
        min_vals = np.min(landmarks, axis=0)
        max_vals = np.max(landmarks, axis=0)
        return (landmarks - min_vals) / (max_vals - min_vals + 1e-8)
    
    elif method == 'standard':
        mean = np.mean(landmarks, axis=0)
        std = np.std(landmarks, axis=0)
        return (landmarks - mean) / (std + 1e-8)
    
    else:
        return landmarks


def get_frame_fps(video_path):
    """
    الحصول على معدل الإطارات (FPS) للفيديو
    
    Args:
        video_path: مسار ملف الفيديو
        
    Returns:
        float: معدل الإطارات
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_duration(video_path):
    """
    الحصول على مدة الفيديو بالثواني
    
    Args:
        video_path: مسار ملف الفيديو
        
    Returns:
        float: مدة الفيديو بالثواني
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration


# مثال على الاستخدام
if __name__ == "__main__":
    print("Creating project directories...")
    create_directories()
    
    print("\\nUtility functions loaded successfully!")
