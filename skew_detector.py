"""
Document Skew Detection - Versi Sederhana yang Benar-benar Bisa Dijalankan
Implementasi 3 metode: FFT, Projection Profiling, dan Hough Transform dengan Voting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.fft import fft2, fftshift
from skimage import io, filters, transform, color, morphology
from skimage.filters import threshold_otsu, sobel
from skimage.transform import hough_line, hough_line_peaks
import os
import re
import math


class DocumentSkewDetector:
    """Main class untuk deteksi kemiringan dokumen dengan 3 metode dan voting"""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.results = {}
        
    def load_image(self, image_path):
        """Load dan preprocess gambar"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File tidak ditemukan: {image_path}")
            
        # Baca gambar
        img = io.imread(image_path)
        
        # Konversi ke grayscale jika perlu
        if len(img.shape) == 3:
            img_gray = color.rgb2gray(img)
        else:
            img_gray = img
            
        # Binarisasi
        threshold = threshold_otsu(img_gray)
        binary_img = img_gray > threshold
        
        return img, img_gray, binary_img
    
    def method1_fft_detection(self, binary_img):
        """Metode 1: FFT-based Detection"""
        try:
            # FFT 2D
            fft_img = fft2(binary_img.astype(float))
            fft_img_shifted = fftshift(fft_img)
            magnitude_spectrum = np.log(np.abs(fft_img_shifted) + 1)
            
            # Binarisasi spektrum
            threshold = threshold_otsu(magnitude_spectrum)
            binary_fft = magnitude_spectrum > threshold
            
            # Hough Transform
            h, theta, d = hough_line(binary_fft)
            _, angles, dists = hough_line_peaks(h, theta, d, min_distance=20, min_angle=5)
            
            if len(angles) == 0:
                return 0.0, 0.0
            
            # Hitung sudut median
            angle = np.rad2deg(np.median(angles))
            
            # Hitung confidence sederhana
            confidence = min(1.0, len(angles) / 10.0)
            
            return angle, confidence
            
        except Exception as e:
            print(f"Error FFT method: {e}")
            return 0.0, 0.0
    
    def method2_projection_profiling(self, binary_img):
        """Metode 2: Projection Profiling"""
        try:
            best_angle = 0
            max_variance = 0
            
            # Test berbagai sudut
            for angle in np.arange(-15, 16, 0.5):
                # Rotasi gambar
                rotated = rotate(binary_img, angle, reshape=False, mode='constant', cval=0)
                
                # Proyeksi horizontal
                h_projection = np.sum(rotated, axis=1)
                
                # Hitung variance
                variance = np.var(h_projection)
                
                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle
            
            # Hitung confidence berdasarkan variance
            # Normalisasi confidence (0-1)
            confidence = min(1.0, max_variance / (binary_img.shape[1] * 0.1))
            
            return best_angle, confidence
            
        except Exception as e:
            print(f"Error Projection method: {e}")
            return 0.0, 0.0
    
    def method3_hough_transform(self, img_gray):
        """Metode 3: Hough Transform"""
        try:
            # Deteksi tepi
            edges = sobel(img_gray)
            edges_binary = edges > threshold_otsu(edges)
            
            # Hough Transform
            h, theta, d = hough_line(edges_binary)
            _, angles, dists = hough_line_peaks(h, theta, d, min_distance=20, min_angle=5)
            
            if len(angles) == 0:
                return 0.0, 0.0
            
            # Konversi ke derajat
            angles_deg = np.rad2deg(angles)
            
            # Hitung sudut rata-rata
            angle = np.mean(angles_deg)
            
            # Confidence berdasarkan jumlah garis
            confidence = min(1.0, len(angles) / 20.0)
            
            return angle, confidence
            
        except Exception as e:
            print(f"Error Hough method: {e}")
            return 0.0, 0.0
    
    def voting_system(self, results):
        """Sistem voting untuk memilih hasil terbaik"""
        valid_results = [(method, angle, conf) for method, angle, conf in results if conf >= self.confidence_threshold]
        
        if not valid_results:
            return 0.0, "none", 0.0
        
        # Best-first voting (pilih confidence tertinggi)
        best_result = max(valid_results, key=lambda x: x[2])
        best_angle = best_result[1]
        best_method = best_result[0]
        best_confidence = best_result[2]
        
        # Weighted average voting
        total_weight = sum(conf for _, _, conf in valid_results)
        weighted_angle = sum(angle * conf for _, angle, conf in valid_results) / total_weight
        
        # Pilih yang lebih konservatif (weighted average jika beda tidak terlalu jauh)
        if abs(best_angle - weighted_angle) < 2.0:
            final_angle = weighted_angle
            final_method = "weighted_average"
        else:
            final_angle = best_angle
            final_method = f"best_first_{best_method}"
        
        return final_angle, final_method, best_confidence
    
    def extract_ground_truth(self, filename):
        """Ekstrak ground truth dari nama file"""
        match = re.search(r'Image-\d+-(-?\d+)', filename)
        if match:
            try:
                return int(match.group(1))
            except:
                return None
        return None
    
    def detect_skew(self, image_path, visualize=False):
        """Main function untuk deteksi kemiringan"""
        print(f"\n🔍 Memproses: {os.path.basename(image_path)}")
        
        # Load gambar
        img, img_gray, binary_img = self.load_image(image_path)
        print(f"✅ Gambar loaded: {img.shape}")
        
        # Jalankan 3 metode
        print("\n📊 Menjalankan deteksi...")
        
        fft_angle, fft_conf = self.method1_fft_detection(binary_img)
        print(f"   FFT Method: {fft_angle:.2f}° (confidence: {fft_conf:.2f})")
        
        proj_angle, proj_conf = self.method2_projection_profiling(binary_img)
        print(f"   Projection: {proj_angle:.2f}° (confidence: {proj_conf:.2f})")
        
        hough_angle, hough_conf = self.method3_hough_transform(img_gray)
        print(f"   Hough Transform: {hough_angle:.2f}° (confidence: {hough_conf:.2f})")
        
        # Voting
        results = [
            ("FFT", fft_angle, fft_conf),
            ("Projection", proj_angle, proj_conf),
            ("Hough", hough_angle, hough_conf)
        ]
        
        final_angle, final_method, final_confidence = self.voting_system(results)
        
        print(f"\n🗳️  Hasil Voting:")
        print(f"   Sudut akhir: {final_angle:.2f}°")
        print(f"   Metode: {final_method}")
        print(f"   Confidence: {final_confidence:.2f}")
        
        # Evaluasi akurasi jika ada ground truth
        ground_truth = self.extract_ground_truth(os.path.basename(image_path))
        if ground_truth is not None:
            error = abs(final_angle - ground_truth)
            print(f"\n📏 Evaluasi Akurasi:")
            print(f"   Ground Truth: {ground_truth}°")
            print(f"   Error: {error:.2f}°")
            print(f"   Status: {'✅ AKURAT' if error <= 2.0 else '❌ KURANG AKURAT'}")
        
        # Simpan hasil
        self.results = {
            'image_path': image_path,
            'methods': {
                'fft': {'angle': fft_angle, 'confidence': fft_conf},
                'projection': {'angle': proj_angle, 'confidence': proj_conf},
                'hough': {'angle': hough_angle, 'confidence': hough_conf}
            },
            'final_angle': final_angle,
            'final_method': final_method,
            'final_confidence': final_confidence,
            'ground_truth': ground_truth
        }
        
        # Visualisasi jika diminta
        if visualize:
            self.visualize_results(img, img_gray, binary_img, final_angle)
        
        return final_angle
    
    def visualize_results(self, img, img_gray, binary_img, angle):
        """Visualisasi hasil"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0,0].imshow(img, cmap='gray')
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Binary image
        axes[0,1].imshow(binary_img, cmap='gray')
        axes[0,1].set_title('Binary Image')
        axes[0,1].axis('off')
        
        # Corrected image
        corrected = rotate(img, -angle, reshape=True, mode='constant', cval=255)
        axes[1,0].imshow(corrected, cmap='gray')
        axes[1,0].set_title(f'Corrected ({-angle:.2f}°)')
        axes[1,0].axis('off')
        
        # Results summary
        axes[1,1].text(0.1, 0.7, f"Final Angle: {angle:.2f}°", fontsize=12)
        axes[1,1].text(0.1, 0.5, f"Method: {self.results['final_method']}", fontsize=12)
        axes[1,1].text(0.1, 0.3, f"Confidence: {self.results['final_confidence']:.2f}", fontsize=12)
        axes[1,1].set_title('Detection Results')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def correct_image(self, image_path, output_path=None, angle=None):
        """Koreksi gambar dan simpan hasil"""
        img, _, _ = self.load_image(image_path)
        
        if angle is None:
            if not self.results:
                raise ValueError("Jalankan detect_skew() terlebih dahulu atau berikan parameter angle")
            angle = self.results['final_angle']
        
        corrected = rotate(img, -angle, reshape=True, mode='constant', cval=255)
        
        if output_path:
            # Convert to uint8 jika perlu
            if corrected.dtype != np.uint8:
                corrected = (corrected * 255).astype(np.uint8)
            io.imsave(output_path, corrected)
            print(f"✅ Gambar terkoreksi disimpan: {output_path}")
        
        return corrected


def main():
    """Function untuk menjalankan deteksi pada sample images"""
    detector = DocumentSkewDetector(confidence_threshold=0.3)
    
    # Cari sample images
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir):
        print(f"❌ Folder {sample_dir} tidak ditemukan")
        return
    
    # Ambil beberapa sample images
    sample_files = []
    for file in os.listdir(sample_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_files.append(os.path.join(sample_dir, file))
            if len(sample_files) >= 3:  # Test maksimal 3 gambar
                break
    
    if not sample_files:
        print(f"❌ Tidak ada file gambar di folder {sample_dir}")
        return
    
    print("🚀 DOCUMENT SKEW DETECTION")
    print("=" * 50)
    
    results_summary = []
    
    for img_path in sample_files:
        try:
            angle = detector.detect_skew(img_path, visualize=False)
            
            # Simpan gambar hasil koreksi
            output_path = f"results/corrected_{os.path.basename(img_path)}"
            os.makedirs("results", exist_ok=True)
            detector.correct_image(img_path, output_path)
            
            results_summary.append({
                'file': os.path.basename(img_path),
                'angle': angle,
                'method': detector.results['final_method'],
                'confidence': detector.results['final_confidence']
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
        
        print("-" * 50)
    
    # Summary
    print("\n📊 RINGKASAN HASIL:")
    print("=" * 50)
    for result in results_summary:
        print(f"📄 {result['file']:<25} | {result['angle']:>6.1f}° | {result['method']:<15} | {result['confidence']:.2f}")
    
    print(f"\n✅ Selesai! Gambar hasil koreksi disimpan di folder 'results/'")


if __name__ == "__main__":
    main()