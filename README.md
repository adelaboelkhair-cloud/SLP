# Sign Language Production (SLP)

## وصف المشروع
تطبيق ذكي لتعرف الإشارات اليدوية (Sign Language) وتحويلها إلى نصوص باستخدام تقنيات الذكاء الاصطناعي و MediaPipe.

## المميزات الرئيسية
- 🎯 التعرف على إشارات اليد في الوقت الفعلي
- 📊 دعم لغات متعددة
- 🎬 معالجة الفيديو والكاميرا المباشرة
- 🧠 نموذج تعلم عميق مدرب
- 📱 واجهة مستخدم سهلة الاستخدام
- 💾 حفظ واستخراج البيانات

## البنية
```
SLP/
├── data/                    # بيانات التدريب
├── models/                  # النماذج المدربة
├── src/
│   ├── mediapipe_processor.py   # معالج MediaPipe
│   ├── model.py                 # نموذج التصنيف
│   ├── gesture_recognizer.py    # معرف الإشارات
│   └── utils.py                 # دوال مساعدة
├── app/                     # التطبيق الرئيسي
│   ├── app.py              # تطبيق PyQt5
│   ├── camera_handler.py   # معالج الكاميرا
│   └── ui/                 # واجهة المستخدم
├── config.py               # إعدادات المشروع
├── requirements.txt        # المتطلبات
└── setup.py               # ملف التثبيت
```

## التثبيت
```bash
git clone https://github.com/adelaboelkhair-cloud/SLP.git
cd SLP
pip install -r requirements.txt
```

## الاستخدام
```bash
python app/app.py
```

## الفريق
مطورو Sign Language Production
