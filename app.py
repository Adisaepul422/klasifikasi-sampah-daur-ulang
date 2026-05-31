"""
FLASK WEB APPLICATION - KLASIFIKASI SAMPAH DENGAN RANDOM FOREST
Aplikasi web untuk mengklasifikasikan gambar sampah organik dan anorganik
"""

from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import pandas as pd
import joblib
import os
import uuid
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Inisialisasi Flask app
app = Flask(__name__)

# Konfigurasi
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Buat folder upload jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model dan scaler
print("="*50)
print("MEMUAT MODEL RANDOM FOREST...")
print("="*50)

try:
    model = joblib.load('random_forest_waste_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model dan Scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    print("Pastikan file 'random_forest_waste_model.pkl' dan 'scaler.pkl' ada!")
    model = None
    scaler = None

# Daftar fitur yang digunakan model
FEATURE_NAMES = [
    'mean_red', 'mean_green', 'mean_blue', 'std_intensity', 'entropy',
    'area_pixels', 'edge_density', 'homogeneity', 'contrast', 'correlation'
]

def allowed_file(filename):
    """Cek apakah ekstensi file diperbolehkan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_features_from_image(image_path):
    """
    Ekstraksi fitur dari gambar untuk diprediksi oleh model Random Forest
    """
    try:
        # Baca gambar dengan OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Gambar tidak dapat dibaca")
        
        # Konversi ke RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Fitur Warna (Mean RGB)
        mean_red = np.mean(img_rgb[:, :, 0])
        mean_green = np.mean(img_rgb[:, :, 1])
        mean_blue = np.mean(img_rgb[:, :, 2])
        
        # 2. Standar Deviasi Intensitas (Grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std_intensity = np.std(gray)
        
        # 3. Entropy (ukuran kekacauan tekstur)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        
        # 4. Area Pixels (dengan thresholding)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        area_pixels = np.sum(thresh > 0)
        
        # 5. Edge Density (kepadatan tepi)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 6. Homogeneity, Contrast, Correlation (dari GLCM sederhana)
        # GLCM (Gray Level Co-occurrence Matrix) sederhana untuk 4 arah
        def compute_glcm_features(gray_img):
            # Resize untuk efisiensi
            gray_img = cv2.resize(gray_img, (128, 128))
            h, w = gray_img.shape
            
            # Inisialisasi GLCM untuk jarak 1 pixel
            glcm = np.zeros((256, 256), dtype=np.float32)
            
            # Hitung co-occurrence (arah horizontal kanan)
            for i in range(h):
                for j in range(w-1):
                    a = gray_img[i, j]
                    b = gray_img[i, j+1]
                    glcm[a, b] += 1
            
            # Normalisasi
            glcm = glcm / glcm.sum()
            
            # Homogeneity
            homogeneity = 0
            contrast = 0
            correlation_numerator = 0
            
            for i in range(256):
                for j in range(256):
                    homogeneity += glcm[i, j] / (1 + abs(i - j))
                    contrast += glcm[i, j] * ((i - j) ** 2)
                    correlation_numerator += (i - np.mean(gray_img)) * (j - np.mean(gray_img)) * glcm[i, j]
            
            # Correlation
            std_i = np.std(gray_img)
            std_j = np.std(gray_img)
            correlation = correlation_numerator / (std_i * std_j) if (std_i * std_j) > 0 else 0
            
            return homogeneity, contrast, correlation
        
        homogeneity, contrast, correlation = compute_glcm_features(gray)
        
        # Kumpulkan semua fitur
        features = {
            'mean_red': mean_red,
            'mean_green': mean_green,
            'mean_blue': mean_blue,
            'std_intensity': std_intensity,
            'entropy': entropy,
            'area_pixels': area_pixels,
            'edge_density': edge_density,
            'homogeneity': homogeneity,
            'contrast': contrast,
            'correlation': correlation
        }
        
        return features
    
    except Exception as e:
        print(f"Error ekstraksi fitur: {e}")
        # Return fitur default jika error
        return {
            'mean_red': 128, 'mean_green': 128, 'mean_blue': 128,
            'std_intensity': 50, 'entropy': 6.5, 'area_pixels': 20000,
            'edge_density': 0.2, 'homogeneity': 0.7, 'contrast': 0.4,
            'correlation': 0.8
        }

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk prediksi gambar
    Menerima upload gambar, ekstrak fitur, dan return hasil klasifikasi
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model belum dimuat. Silakan training model terlebih dahulu.'}), 500
    
    # Cek apakah ada file yang diupload
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'File tidak dipilih'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Format file tidak didukung. Gunakan: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
    
    try:
        # Simpan file sementara
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Ekstraksi fitur dari gambar
        features = extract_features_from_image(filepath)
        
        # Buat feature vector sesuai urutan
        feature_vector = np.array([[
            features['mean_red'],
            features['mean_green'],
            features['mean_blue'],
            features['std_intensity'],
            features['entropy'],
            features['area_pixels'],
            features['edge_density'],
            features['homogeneity'],
            features['contrast'],
            features['correlation']
        ]])
        
        # Normalisasi fitur
        feature_scaled = scaler.transform(feature_vector)
        
        # Prediksi
        prediction = model.predict(feature_scaled)[0]
        probabilities = model.predict_proba(feature_scaled)[0]
        
        # Hasil prediksi
        class_name = "Recyclable (Anorganik)" if prediction == 1 else "Organik"
        suggestion = "♻️ Dapat didaur ulang menjadi produk baru seperti botol, kertas, atau kemasan" if prediction == 1 else "🍂 Dapat diolah menjadi kompos atau pupuk organik"
        recycling_tips = {
            0: "• Potong kecil-kecil untuk mempercepat pengomposan\n• Campur dengan daun kering\n• Hindari mencampur dengan sampah plastik",
            1: "• Bersihkan dari sisa makanan\n• Pisahkan berdasarkan jenis (plastik, kertas, logam, kaca)\n• Lipat atau tekan agar lebih hemat tempat"
        }
        
        result = {
            'success': True,
            'prediction': int(prediction),
            'class_name': class_name,
            'confidence': {
                'organic': float(probabilities[0]),
                'recyclable': float(probabilities[1])
            },
            'confidence_percentage': {
                'organic': f"{probabilities[0]*100:.1f}%",
                'recyclable': f"{probabilities[1]*100:.1f}%"
            },
            'suggestion': suggestion,
            'recycling_tips': recycling_tips[prediction],
            'image_path': filepath,
            'image_url': f"/static/uploads/{unique_filename}",
            'features': {k: f"{v:.2f}" for k, v in features.items()}
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error dalam prediksi: {str(e)}")
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint untuk cek status server"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 APLIKASI KLASIFIKASI SAMPAH")
    print("="*50)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Model loaded: {model is not None}")
    print(f"Scaler loaded: {scaler is not None}")
    print("\n📍 Akses aplikasi di: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)