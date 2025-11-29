"""
Document Skew Detection Web App
Streamlit interface untuk deteksi dan koreksi kemiringan dokumen
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.fft import fft2, fftshift
from skimage import io, filters, transform, color
from skimage.filters import threshold_otsu, sobel
from skimage.transform import hough_line, hough_line_peaks
from PIL import Image
import tempfile
import os
import re
import math

# Page config
st.set_page_config(
    page_title="Document Skew Detection",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentSkewDetector:
    """Document skew detector dengan 3 metode"""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.results = {}
    
    def preprocess_image(self, img):
        """Preprocess gambar untuk deteksi"""
        # Handle RGBA images (4 channels)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # Remove alpha channel
        
        # Konversi ke grayscale
        if len(img.shape) == 3:
            img_gray = color.rgb2gray(img)
        else:
            img_gray = img
        
        # Binarisasi
        threshold = threshold_otsu(img_gray)
        binary_img = img_gray > threshold
        
        return img_gray, binary_img
    
    def method1_fft(self, img_gray):
        """FFT-based skew detection - Original notebook implementation"""
        try:
            from skimage import exposure
            from skimage.util import img_as_ubyte
            
            # Transformasi Fourier 2D untuk domain frekuensi
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))
            
            # Normalisasi kontras citra FFT
            stretched = exposure.rescale_intensity(magnitude, out_range=(0, 1))
            
            # Equalisasi histogram untuk meningkatkan distribusi
            eq = exposure.equalize_hist(stretched)
            
            # Ubah ke 8-bit untuk binarisasi
            eq = img_as_ubyte(eq)
            
            # Binarisasi dengan Otsu pada hasil equalize
            otsu_fft = threshold_otsu(eq)
            binary_fft = eq > otsu_fft
            
            # Transformasi Hough untuk deteksi garis
            h, theta, d = hough_line(binary_fft)
            _, angles, dists = hough_line_peaks(h, theta, d, min_distance=20, min_angle=5)
            
            # Saring sudut yang garisnya melalui bagian tengah gambar
            center = np.array(binary_fft.shape) / 2
            valid_angles = []
            
            for angle, dist in zip(angles, dists):
                # Hitung seberapa jauh garis dari pusat
                dist_from_center = abs(dist - (center[1]*np.cos(angle) + center[0]*np.sin(angle)))
                if dist_from_center < 20:  # Ambang toleransi
                    valid_angles.append(angle)
            
            # Hitung sudut kemiringan median dari garis-garis valid
            if valid_angles:
                angle_fft = np.rad2deg(np.median(valid_angles))
                
                # Hitung confidence berdasarkan kekuatan garis
                try:
                    h_peaks = hough_line_peaks(h, theta, d, threshold=0.5*np.max(h))
                    line_strengths = [
                        h[int((np.rad2deg(a) % 180)), int(d)]
                        for a, d in zip(h_peaks[1], h_peaks[2])
                        if 0 <= int((np.rad2deg(a) % 180)) < h.shape[0] and
                           0 <= int(d) < h.shape[1]
                    ]
                    if line_strengths:
                        line_strength = np.mean(line_strengths)
                        confidence_fft = min(1.0, line_strength / np.max(h))
                    else:
                        confidence_fft = 0.5
                except:
                    confidence_fft = 0.5
            else:
                angle_fft = 0.0
                confidence_fft = 0.0
                
            return angle_fft, confidence_fft
        except Exception as e:
            return 0.0, 0.0
    
    def method2_projection(self, binary_img):
        """Projection profiling with connected components - Original notebook implementation"""
        try:
            from skimage import measure
            
            # Deteksi komponen terhubung (connected components)
            label_img = measure.label(binary_img)
            regions = measure.regionprops(label_img)
            
            if len(regions) == 0:
                return 0.0, 0.0
            
            # Hitung rata-rata lebar dan tinggi AABB (bounding box)
            avg_width = np.mean([r.bbox[3] - r.bbox[1] for r in regions])
            avg_height = np.mean([r.bbox[2] - r.bbox[0] for r in regions])
            
            # Ambil titik tengah atas dan bawah dari setiap komponen
            top_points, bottom_points = [], []
            for r in regions:
                width = r.bbox[3] - r.bbox[1]
                height = r.bbox[2] - r.bbox[0]
                
                # Filter: hanya komponen yang berukuran wajar
                if width < 5 * avg_width and height < 5 * avg_height:
                    x_center = (r.bbox[1] + r.bbox[3]) / 2
                    top_points.append([x_center, r.bbox[0]])
                    bottom_points.append([x_center, r.bbox[2]])
            
            if len(top_points) < 3:  # Minimal 3 titik untuk analisis
                return 0.0, 0.0
            
            # Cari sudut terbaik dengan memutar titik-titik
            variances = []
            best_angle = 0
            max_var = 0
            
            test_angles = np.linspace(-90, 90, 181)
            center = np.array(binary_img.shape[::-1]) / 2  # Pusat gambar
            
            for angle in test_angles:
                rad = np.deg2rad(angle)
                rot_mat = np.array([[np.cos(rad), -np.sin(rad)],
                                    [np.sin(rad),  np.cos(rad)]])
                
                # Putar titik-titik terhadap pusat gambar
                rotated_top = (np.array(top_points) - center) @ rot_mat.T + center
                rotated_bottom = (np.array(bottom_points) - center) @ rot_mat.T + center
                
                # Buat histogram posisi vertikal
                hist_top, _ = np.histogram(rotated_top[:,1], bins=200, range=(0, binary_img.shape[0]))
                hist_bottom, _ = np.histogram(rotated_bottom[:,1], bins=200, range=(0, binary_img.shape[0]))
                
                # Hitung variansi total
                var_top = np.var(hist_top)
                var_bottom = np.var(hist_bottom)
                total_var = var_top + var_bottom
                variances.append(total_var)
                
                if total_var > max_var:
                    max_var = total_var
                    best_angle = angle
            
            # Hitung confidence berdasarkan variansi
            mean_var = np.mean(variances)
            std_var = np.std(variances)
            confidence_proj = min(1.0, (max_var - mean_var) / (std_var + 1e-6))
            
            return best_angle, confidence_proj
        except Exception as e:
            return 0.0, 0.0
    
    def method3_hough(self, img_gray):
        """Spatial Domain Lines (Hough-based) - Original notebook implementation"""
        try:
            from skimage.util import img_as_ubyte
            
            # 1. Deteksi tepi menggunakan filter Sobel
            edges = sobel(img_gray)
            edges = img_as_ubyte(edges)
            
            # Binarisasi tepi menggunakan threshold Otsu
            edges_binary = edges > threshold_otsu(edges)
            
            # 2. Transformasi Hough
            h, theta, d = hough_line(edges_binary)
            _, angles, dists = hough_line_peaks(h, theta, d, min_distance=20, min_angle=5)
            
            if len(angles) == 0:
                return 0.0, 0.0
            
            angles_deg = np.rad2deg(angles)
            
            # 3. Pengelompokkan garis menjadi set-set paralel
            parallel_sets = []
            for angle, dist in zip(angles_deg, dists):
                matched = False
                for s in parallel_sets:
                    # Cek apakah sudut mirip (toleransi 5 derajat)
                    if abs(s[0][0] - angle) < 5:
                        s.append((angle, dist))
                        matched = True
                        break
                if not matched:
                    parallel_sets.append([(angle, dist)])
            
            # Hitung sudut representatif untuk setiap kelompok
            set_angles = []
            set_lengths = []
            for s in parallel_sets:
                avg_angle = np.mean([a for a, _ in s])
                set_angles.append(avg_angle)
                set_lengths.append(len(s))
            
            if len(set_angles) == 0:
                return 0.0, 0.0
            
            # Urutkan kelompok berdasarkan ukuran (jumlah garis)
            sorted_sets = sorted(zip(set_angles, set_lengths), key=lambda x: -x[1])
            
            # Kasus 1: Hanya satu kelompok
            if len(sorted_sets) == 1:
                angle_hough = sorted_sets[0][0]
                confidence_hough = min(1.0, sorted_sets[0][1] / 20)
            # Kasus 2: Beberapa kelompok, cari pasangan tegak lurus
            else:
                main_angle = sorted_sets[0][0]
                
                # Cari sudut yang paling tegak lurus
                perpendicular_diff = []
                for angle, length in sorted_sets[1:]:
                    diff = abs(abs(angle - main_angle) - 90)
                    perpendicular_diff.append((diff, angle, length))
                
                if perpendicular_diff:
                    best_diff, best_angle, best_length = min(perpendicular_diff, key=lambda x: x[0])
                    
                    if best_diff < 15:  # Toleransi 15° untuk tegak lurus
                        # Rata-rata berbobot
                        total_weight = sorted_sets[0][1] + best_length
                        angle_hough = (main_angle * sorted_sets[0][1] + best_angle * best_length) / total_weight
                        confidence_hough = min(1.0, (sorted_sets[0][1] + best_length) / 30 * (1 - best_diff/90))
                    else:
                        angle_hough = main_angle
                        confidence_hough = min(1.0, sorted_sets[0][1] / 20)
                else:
                    angle_hough = main_angle
                    confidence_hough = min(1.0, sorted_sets[0][1] / 20)
            
            return angle_hough, confidence_hough
        except Exception as e:
            return 0.0, 0.0
    
    def voting_system(self, results):
        """Voting system - Original notebook implementation"""
        # Seleksi awal kandidat dengan confidence >= 0.5
        candidates = []
        for method, angle, conf in results:
            if conf >= 0.5:  # Threshold confidence sesuai paper
                candidates.append((method, angle, conf))
        
        # Jika tidak ada kandidat yang memenuhi syarat
        if not candidates:
            return 0.0, 'None', 0.0
        
        # Best-First Voting (pilih dengan confidence tertinggi)
        best_first = max(candidates, key=lambda x: x[2])
        final_angle_best = best_first[1]
        best_method = best_first[0]
        final_confidence = best_first[2]
        
        return final_angle_best, best_method, final_confidence
    
    def detect_skew(self, img_array):
        """Main detection function"""
        img_gray, binary_img = self.preprocess_image(img_array)
        
        # Jalankan 3 metode
        fft_angle, fft_conf = self.method1_fft(img_gray)  # FFT uses grayscale
        proj_angle, proj_conf = self.method2_projection(binary_img)  # Projection uses binary
        hough_angle, hough_conf = self.method3_hough(img_gray)  # Hough uses grayscale
        
        # Voting
        results = [
            ("FFT", fft_angle, fft_conf),
            ("Projection", proj_angle, proj_conf),
            ("Hough", hough_angle, hough_conf)
        ]
        
        final_angle, final_method, final_confidence = self.voting_system(results)
        
        self.results = {
            'methods': {
                'FFT': {'angle': fft_angle, 'confidence': fft_conf},
                'Projection': {'angle': proj_angle, 'confidence': proj_conf},
                'Hough': {'angle': hough_angle, 'confidence': hough_conf}
            },
            'final_angle': final_angle,
            'final_method': final_method,
            'final_confidence': final_confidence
        }
        
        return final_angle, img_gray, binary_img
    
    def correct_image(self, img_array, angle):
        """Koreksi gambar"""
        return rotate(img_array, -angle, reshape=True, mode='constant', cval=255)

# Initialize detector
@st.cache_resource
def get_detector():
    return DocumentSkewDetector(confidence_threshold=0.3)

def main():
    # Header
    st.title("📄 Document Skew Detection & Correction")
    st.markdown("Deteksi dan koreksi kemiringan dokumen menggunakan 3 metode: FFT, Projection Profiling, dan Hough Transform")
    
    # Sidebar
    st.sidebar.header("⚙️ Pengaturan")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.1)
    show_intermediate = st.sidebar.checkbox("Tampilkan Hasil Intermediate", value=True)
    
    detector = get_detector()
    detector.confidence_threshold = confidence_threshold
    
    # File upload
    st.header("📁 Upload Gambar Dokumen")
    uploaded_file = st.file_uploader(
        "Pilih gambar dokumen yang ingin dideteksi kemiringannya",
        type=['png', 'jpg', 'jpeg'],
        help="Format yang didukung: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display original image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Gambar Asli")
            st.image(image, caption=f"Size: {img_array.shape}", width="stretch")
        
        # Process button
        if st.button("🔍 Deteksi Kemiringan", type="primary"):
            with st.spinner("Memproses gambar..."):
                # Detect skew
                final_angle, img_gray, binary_img = detector.detect_skew(img_array)
                
                # Correct image
                corrected_img = detector.correct_image(img_array, final_angle)
                
            # Display corrected image
            with col2:
                st.subheader("✅ Gambar Terkoreksi")
                st.image(corrected_img, caption=f"Dirotasi {-final_angle:.1f}°", width="stretch")
            
            # Results section
            st.header("📊 Hasil Deteksi")
            
            # Main results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Sudut Kemiringan", f"{final_angle:.2f}°")
            
            with col2:
                st.metric("🏆 Metode Terpilih", detector.results['final_method'])
            
            with col3:
                st.metric("📈 Confidence", f"{detector.results['final_confidence']:.2f}")
            
            # Method comparison
            if show_intermediate:
                st.subheader("🔬 Perbandingan Metode")
                
                methods_data = []
                for method, data in detector.results['methods'].items():
                    methods_data.append({
                        'Metode': method,
                        'Sudut (°)': f"{data['angle']:.2f}",
                        'Confidence': f"{data['confidence']:.2f}",
                        'Status': '✅ Valid' if data['confidence'] >= confidence_threshold else '❌ Invalid'
                    })
                
                st.table(methods_data)
                
                # Visualization
                st.subheader("👁️ Visualisasi Proses")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original grayscale
                axes[0].imshow(img_gray, cmap='gray')
                axes[0].set_title('Grayscale')
                axes[0].axis('off')
                
                # Binary
                axes[1].imshow(binary_img, cmap='gray')
                axes[1].set_title('Binary (untuk FFT & Projection)')
                axes[1].axis('off')
                
                # Corrected
                axes[2].imshow(corrected_img, cmap='gray' if len(corrected_img.shape)==2 else None)
                axes[2].set_title(f'Hasil Koreksi ({-final_angle:.1f}°)')
                axes[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Download section
            st.header("💾 Download Hasil")
            
            # Convert corrected image for download
            if corrected_img.dtype != np.uint8:
                corrected_img_uint8 = (corrected_img * 255).astype(np.uint8)
            else:
                corrected_img_uint8 = corrected_img
            
            corrected_pil = Image.fromarray(corrected_img_uint8)
            
            # Save to bytes
            import io
            buf = io.BytesIO()
            corrected_pil.save(buf, format='PNG')
            byte_data = buf.getvalue()
            
            st.download_button(
                label="📥 Download Gambar Terkoreksi",
                data=byte_data,
                file_name=f"corrected_{uploaded_file.name}",
                mime="image/png"
            )
            
            # Success message
            st.success(f"✅ Deteksi selesai! Kemiringan {final_angle:.2f}° berhasil dikoreksi menggunakan metode {detector.results['final_method']}")
    
    else:
        # Demo section
        st.header("🎯 Demo")
        st.info("Upload gambar dokumen di atas untuk mulai deteksi kemiringan")
        
        # Instructions
        with st.expander("📖 Cara Penggunaan"):
            st.markdown("""
            1. **Upload gambar** dokumen yang ingin dideteksi kemiringannya
            2. **Klik tombol "Deteksi Kemiringan"** untuk memproses
            3. **Lihat hasil** deteksi dari 3 metode berbeda
            4. **Download gambar** yang sudah dikoreksi
            
            ### Metode Deteksi:
            - **FFT**: Analisis spektrum frekuensi
            - **Projection**: Analisis proyeksi horizontal dengan rotasi
            - **Hough Transform**: Deteksi garis dominan
            
            ### Tips:
            - Gambar dengan teks yang jelas akan memberikan hasil lebih akurat
            - Confidence threshold yang lebih rendah akan menerima lebih banyak kandidat
            """)
        
        # Sample images info
        with st.expander("🖼️ Contoh Gambar yang Cocok"):
            st.markdown("""
            **Gambar yang baik untuk deteksi:**
            - Dokumen dengan teks yang jelas
            - Scan dokumen yang miring
            - Foto dokumen dengan pencahayaan cukup
            - Format PNG, JPG, atau JPEG
            
            **Hindari:**
            - Gambar dengan noise berlebihan
            - Dokumen dengan teks sangat kecil
            - Gambar dengan kualitas sangat rendah
            """)

if __name__ == "__main__":
    main()