"""
Collect Data Script - Quick script to start data collection
سكريبت جمع البيانات السريع
"""

from src.data_collector import DataCollector
from config import GESTURES

if __name__ == "__main__":
    print("Sign Language Production - Data Collection")
    print("="*50)
    print(f"Available gestures: {', '.join(GESTURES)}")
    print("="*50)
    
    collector = DataCollector()
    
    # عرض الإحصائيات الحالية
    print("\nCurrent data statistics:")
    stats = collector.get_statistics()
    total = 0
    for gesture, count in stats.items():
        print(f"  {gesture}: {count} samples")
        total += count
    print(f"  Total: {total} samples")
    
    # بدء جمع البيانات
    print("\nStarting data collection...")
    print("Press Enter to start collecting data for each gesture")
    
    collector.collect_all_gestures()
    
    # عرض الإحصائيات النهائية
    print("\nFinal data statistics:")
    stats = collector.get_statistics()
    total = 0
    for gesture, count in stats.items():
        print(f"  {gesture}: {count} samples")
        total += count
    print(f"  Total: {total} samples")
    
    collector.close()
