# ♻️ WasteWise - Klasifikasi Sampah dengan Random Forest

Aplikasi web untuk mengklasifikasikan gambar sampah menjadi **Organik** atau **Recyclable (Anorganik)** menggunakan algoritma **Random Forest**.

## 📋 Deskripsi Proyek

Proyek ini dibuat untuk memenuhi tugas mata kuliah Artificial Intelligence. Aplikasi menggunakan model Random Forest yang dilatih dengan 100 pohon keputusan untuk mengekstraksi fitur dari gambar (warna, tekstur, bentuk) dan menentukan jenis sampah.

## 🚀 Fitur

- Upload gambar sampah (drag & drop atau pilih file)
- Ekstraksi fitur otomatis dari gambar (RGB, entropy, edge density, GLCM)
- Prediksi real-time dengan model Random Forest
- Tampilan confidence level untuk setiap kelas
- Rekomendasi pengelolaan sampah berdasarkan hasil prediksi

## 🛠️ Teknologi yang Digunakan

| Technology | Usage |
|------------|-------|
| Python 3.10+ | Backend |
| Flask | Web Framework |
| Scikit-Learn | Random Forest Model |
| OpenCV | Image Processing |
| Bootstrap 5 | Frontend UI |

## 📂 Struktur Proyek
