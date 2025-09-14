Résidanat QCM Simulator - Prototype
==================================

هذا أرشيف يحتوي على prototype لتطبيق Streamlit يقوم بتوليد ومحاكاة QCM، تسجيل النتائج، وdashboard تحليلي.
ملاحظة: يجب إدخال مفتاح OpenAI API لتمكين وظائف التوليد والتصحيح والتحليلات النصية.

ملفّات:
- app.py: التطبيق الرئيسي (Streamlit)
- requirements.txt: الحزم المطلوبة
- README.md: هذه التعليمات

تشغيل سريع:
1. تثبيت الحزم: `pip install -r requirements.txt`
2. ضبط المتغير البيئي OPENAI_API_KEY (اختياري لكن مطلوب للـAI)
3. تشغيل: `streamlit run app.py`