"""
MODEL TRAINING - RANDOM FOREST UNTUK KLASIFIKASI SAMPAH
File ini dijalankan SEKALI untuk membuat model yang akan digunakan di app.py
"""

# Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setting style untuk plot
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

print("="*60)
print("TRAINING MODEL RANDOM FOREST UNTUK KLASIFIKASI SAMPAH")
print("="*60)

# 1. Load Dataset
print("\n[1] Memuat dataset...")
df = pd.read_csv('waste_dataset.csv')
print(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# 2. Eksplorasi Dataset
print("\n[2] Eksplorasi Dataset...")
print("\n5 Data Pertama:")
print(df.head())
print("\nInfo Dataset:")
print(df.info())
print("\nStatistik Deskriptif:")
print(df.describe())
print("\nDistribusi Kelas:")
class_dist = df['class'].value_counts()
print(f"Organik (0): {class_dist[0]} sampel")
print(f"Recyclable (1): {class_dist[1]} sampel")

# Visualisasi distribusi kelas
plt.figure(figsize=(8,5))
sns.countplot(x='class', data=df)
plt.title('Distribusi Kelas Sampah', fontsize=14)
plt.xlabel('Kelas (0=Organik, 1=Recyclable)')
plt.ylabel('Jumlah Sampel')
for i, v in enumerate(class_dist):
    plt.text(i, v + 1, str(v), ha='center', fontsize=12)
plt.savefig('static/class_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

# 3. Preprocessing Data
print("\n[3] Preprocessing Data...")
X = df.drop(columns=['image_id', 'class'])
y = df['class']
feature_names = X.columns.tolist()
print(f"Fitur yang digunakan: {feature_names}")
print(f"Jumlah fitur: {len(feature_names)}")

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Normalisasi data selesai")

# 4. Split Data
print("\n[4] Membagi data training dan testing...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data training: {X_train.shape[0]} sampel")
print(f"Data testing: {X_test.shape[0]} sampel")

# 5. Training Random Forest
print("\n[5] Melatih model Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,       # Jumlah pohon keputusan
    max_depth=10,           # Kedalaman maksimal pohon
    min_samples_split=5,    # Minimal sampel untuk split
    min_samples_leaf=2,     # Minimal sampel di leaf
    random_state=42,
    n_jobs=-1               # Gunakan semua processor
)

rf_model.fit(X_train, y_train)
print("Model berhasil dilatih!")

# 6. Evaluasi Model
print("\n[6] Evaluasi Model...")
y_pred = rf_model.predict(X_test)

# Metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*50)
print("HASIL EVALUASI MODEL")
print("="*50)
print(f"Akurasi  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5)
print(f"\nCross-Validation (5-fold):")
print(f"Rata-rata CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Classification Report
print("\nLaporan Klasifikasi Lengkap:")
print(classification_report(y_test, y_pred, target_names=['Organik (0)', 'Recyclable (1)']))

# 7. Confusion Matrix
print("\n[7] Membuat Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Organik', 'Recyclable'],
            yticklabels=['Organik', 'Recyclable'])
plt.title('Confusion Matrix - Klasifikasi Sampah', fontsize=14)
plt.xlabel('Prediksi', fontsize=12)
plt.ylabel('Aktual', fontsize=12)
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("Confusion Matrix tersimpan di 'static/confusion_matrix.png'")

# 8. Feature Importance
print("\n[8] Analisis Feature Importance...")
feature_importance = pd.DataFrame({
    'Fitur': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTingkat Kepentingan Fitur:")
print("="*40)
for i, row in feature_importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"{row['Fitur']:20s}: {row['Importance']:.4f} {bar}")

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Fitur', data=feature_importance, palette='viridis')
plt.title('Feature Importance - Random Forest', fontsize=14)
plt.xlabel('Nilai Kepentingan', fontsize=12)
plt.tight_layout()
plt.savefig('static/feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nFeature Importance plot tersimpan di 'static/feature_importance.png'")

# 9. Simpan Model
print("\n[9] Menyimpan model...")
joblib.dump(rf_model, 'random_forest_waste_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model berhasil disimpan sebagai 'random_forest_waste_model.pkl'")
print("✅ Scaler berhasil disimpan sebagai 'scaler.pkl'")

# 10. Ringkasan Final
print("\n" + "="*60)
print("RINGKASAN TRAINING MODEL")
print("="*60)
print(f"""
| Metrik               | Nilai                     |
|----------------------|---------------------------|
| Dataset              | {df.shape[0]} sampel, {len(feature_names)} fitur |
| Training Size        | {X_train.shape[0]} sampel ({X_train.shape[0]/df.shape[0]*100:.0f}%) |
| Testing Size         | {X_test.shape[0]} sampel ({X_test.shape[0]/df.shape[0]*100:.0f}%) |
| Akurasi              | {accuracy:.4f} ({accuracy*100:.2f}%) |
| Precision            | {precision:.4f}           |
| Recall               | {recall:.4f}              |
| F1-Score             | {f1:.4f}                  |
| Cross Validation (5-fold) | {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}) |
| N_Estimators         | 100                       |
| Max Depth            | 10                        |
| Random State         | 42                        |
""")

print("\n✅ TRAINING SELESAI! Model siap digunakan di app.py")