# إرشادات رفع المشروع على GitHub

## الخطوات المتبقية:

### 1. إنشاء Repository على GitHub

1. اذهب إلى: https://github.com/new
2. أدخل اسم المشروع (مثلاً: `seniorproject-rag-system`)
3. اختر **Public** أو **Private**
4. **لا** تضع علامة على "Initialize this repository with a README"
5. اضغط **Create repository**

### 2. ربط المشروع المحلي بـ GitHub

بعد إنشاء الـ repository، ستظهر لك تعليمات. استخدم الأوامر التالية:

```bash
# استبدل YOUR_USERNAME و REPO_NAME بالقيم الصحيحة
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

أو إذا كنت تستخدم SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

### 3. التحقق من الرفع

بعد الرفع، اذهب إلى صفحة الـ repository على GitHub للتحقق من أن جميع الملفات تم رفعها بنجاح.

## ملاحظات مهمة:

✅ ملف `.env` **غير مضاف** (آمن - لن يتم رفعه)
✅ جميع ملفات `__pycache__` **غير مضافة** (موجودة في .gitignore)
✅ ملفات النماذج الكبيرة **غير مضافة** (موجودة في .gitignore)

## الملفات المهمة المحفوظة:

✅ `chunks.json` - موجود
✅ `vectors.npy` - موجود
✅ `embeddings/chunks.json` - موجود
✅ `embeddings/vectors.npy` - موجود


