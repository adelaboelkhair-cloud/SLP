"""
Training Script - Train the gesture recognition model
سكريبت التدريب - تدريب نموذج التعرف على الإشارات
"""

import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from src.model import GestureModel
from src.mediapipe_processor import MediaPipeProcessor
from config import TRAINING_CONFIG, GESTURES, MODELS_DIR


def load_data(data_dir="data/"):
    """
    تحميل البيانات من المجلدات
    
    Args:
        data_dir: مجلد البيانات
        
    Returns:
        tuple: (X, y) - البيانات والتسميات
    """
    X = []
    y = []
    
    data_path = Path(data_dir)
    
    for gesture_idx, gesture in enumerate(GESTURES):
        gesture_dir = data_path / gesture
        
        if not gesture_dir.exists():
            print(f"Warning: Directory not found for gesture '{gesture}'")
            continue
        
        files = list(gesture_dir.glob("*.npy"))
        print(f"Loading {len(files)} samples for gesture '{gesture}'...")
        
        for file in files:
            try:
                landmarks = np.load(file)
                
                # تسطيح النقاط
                flat_landmarks = landmarks.flatten()
                
                X.append(flat_landmarks)
                y.append(gesture_idx)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return np.array(X), np.array(y)


def prepare_data(X, y, test_size=None, validation_split=None):
    """
    تجهيز البيانات للتدريب
    
    Args:
        X: البيانات
        y: التسميات
        test_size: نسبة بيانات الاختبار
        validation_split: نسبة بيانات التحقق
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if test_size is None:
        test_size = TRAINING_CONFIG["test_size"]
    if validation_split is None:
        validation_split = TRAINING_CONFIG["validation_split"]
    
    # تحويل التسميات إلى one-hot encoding
    y_onehot = keras.utils.to_categorical(y, len(GESTURES))
    
    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=test_size,
        random_state=42
    )
    
    # تقسيم بيانات التدريب إلى تدريب وتحقق
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=42
    )
    
    # تطبيع البيانات
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, X_val, y_train, y_val, model_path=None):
    """
    تدريب النموذج
    
    Args:
        X_train: بيانات التدريب
        X_val: بيانات التحقق
        y_train: تسميات التدريب
        y_val: تسميات التحقق
        model_path: مسار حفظ النموذج
        
    Returns:
        GestureModel: النموذج المدرب
    """
    print("\\n" + "="*50)
    print("Starting Model Training")
    print("="*50)
    
    # إنشاء النموذج
    model = GestureModel(input_shape=X_train.shape[1])
    
    # عرض ملخص النموذج
    model.get_summary()
    
    # تدريب النموذج
    print("\\nTraining the model...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    # حفظ النموذج
    if model_path is None:
        model_path = MODELS_DIR / "gesture_classifier_v1.h5"
    
    model.save(str(model_path))
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    تقييم النموذج
    
    Args:
        model: النموذج المدرب
        X_test: بيانات الاختبار
        y_test: تسميات الاختبار
        
    Returns:
        dict: نتائج التقييم
    """
    print("\\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    results = model.evaluate(X_test, y_test)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    
    return results


def main():
    """البرنامج الرئيسي للتدريب"""
    print("\\n" + "="*50)
    print("Sign Language Production - Model Training")
    print("="*50)
    
    # تحميل البيانات
    print("\\nLoading data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("Error: No data found! Please collect data first using data_collector.py")
        return
    
    print(f"Total samples: {len(X)}")
    print(f"Data shape: {X.shape}")
    
    # تجهيز البيانات
    print("\\nPreparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # تدريب النموذج
    model, history = train_model(X_train, X_val, y_train, y_val)
    
    # تقييم النموذج
    results = evaluate_model(model, X_test, y_test)
    
    print("\\n" + "="*50)
    print("Training Completed Successfully!")
    print("="*50)
    
    return model, history, results


if __name__ == "__main__":
    main()
