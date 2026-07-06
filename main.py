import os
import base64
import cv2
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import warnings
from mediapipe.python.solutions import face_detection as mp_face_detection
from mediapipe.python.solutions import face_mesh as mp_face_mesh

warnings.filterwarnings('ignore')

# ========================
# KONFIGURASI DAN INISIALISASI
# ========================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# MediaPipe
FACE_DETECTION = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Konstanta untuk ekstraksi
LANDMARK_IMAGE_SIZE = 300
MOUTH_LEFT_INDEX = 61
MOUTH_RIGHT_INDEX = 291
LEFT_EYE_TOP_INDEX = 33
LEFT_EYE_BOTTOM_INDEX = 133
RIGHT_EYE_TOP_INDEX = 362
RIGHT_EYE_BOTTOM_INDEX = 263
CHIN_INDEX = 152
FOREHEAD_INDEX = 1
TOTAL_LANDMARKS = 478

# ========================
# MODEL ARSITEKTUR
# ========================

class StrokeDetectionMLP(nn.Module):
    """Multi-layer perceptron untuk deteksi stroke dari landmark wajah."""
    def __init__(self, input_dim=956, hidden_dims=None, output_dim=2, dropout=0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ========================
# MEMUAT MODEL
# ========================

MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = StrokeDetectionMLP()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print(f"Model file not found at {MODEL_PATH}")

# ========================
# FUNGSI BANTUAN
# ========================

def extract_landmarks_and_crop(image):
    """
    Mendeteksi wajah, melakukan crop, dan mengekstrak 478 titik landmark.
    Mengembalikan fitur (list), gambar crop, bounding box, dan koordinat landmark.
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = FACE_DETECTION.process(rgb)
    if not detections.detections:
        return None, None, None, None

    bbox = detections.detections[0].location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    box_w = int(bbox.width * w)
    box_h = int(bbox.height * h)

    margin_x = int(box_w * 0.15)
    margin_y = int(box_h * 0.15)
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    box_w = min(w - x, box_w + 2 * margin_x)
    box_h = min(h - y, box_h + 2 * margin_y)

    cropped = image[y:y+box_h, x:x+box_w]
    if cropped.size == 0:
        return None, None, None, None

    crop_resized = cv2.resize(cropped, (LANDMARK_IMAGE_SIZE, LANDMARK_IMAGE_SIZE))
    rgb_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    mesh = FACE_MESH.process(rgb_crop)
    if not mesh.multi_face_landmarks:
        return None, None, None, None

    landmarks_raw = mesh.multi_face_landmarks[0].landmark
    # Fitur: koordinat x,y dinormalisasi ke ukuran 300x300
    features = []
    for lm in landmarks_raw:
        features.append(lm.x * LANDMARK_IMAGE_SIZE)
        features.append(lm.y * LANDMARK_IMAGE_SIZE)
    features = np.array(features, dtype=np.float32)

    # Standardisasi
    if features.std() > 1e-6:
        features = (features - features.mean()) / (features.std() + 1e-6)

    landmark_points = [(lm.x * LANDMARK_IMAGE_SIZE, lm.y * LANDMARK_IMAGE_SIZE)
                       for lm in landmarks_raw]

    return features.tolist(), cropped, (x, y, box_w, box_h), landmark_points


def draw_landmarks_on_image(image_bgr, landmark_points, size=(LANDMARK_IMAGE_SIZE, LANDMARK_IMAGE_SIZE)):
    """Menggambar titik landmark hijau pada gambar BGR."""
    img = image_bgr.copy()
    h, w = img.shape[:2]
    scale_x = w / size[0]
    scale_y = h / size[1]
    for (lx, ly) in landmark_points:
        px = int(lx * scale_x)
        py = int(ly * scale_y)
        cv2.circle(img, (px, py), 2, (0, 255, 0), -1)
    return img


def calculate_asymmetry_metrics(landmark_points):
    """
    Menghitung metrik asimetri wajah berdasarkan titik-titik landmark tertentu.
    Mengembalikan dictionary metrik.
    """
    if not landmark_points or len(landmark_points) < TOTAL_LANDMARKS:
        return None

    left_mouth = landmark_points[MOUTH_LEFT_INDEX]
    right_mouth = landmark_points[MOUTH_RIGHT_INDEX]
    left_eye_top = landmark_points[LEFT_EYE_TOP_INDEX]
    left_eye_bottom = landmark_points[LEFT_EYE_BOTTOM_INDEX]
    right_eye_top = landmark_points[RIGHT_EYE_TOP_INDEX]
    right_eye_bottom = landmark_points[RIGHT_EYE_BOTTOM_INDEX]
    chin = landmark_points[CHIN_INDEX]
    forehead = landmark_points[FOREHEAD_INDEX]

    mouth_asymmetry = abs(left_mouth[0] - right_mouth[0]) / LANDMARK_IMAGE_SIZE
    left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
    right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
    eye_asymmetry = abs(left_eye_height - right_eye_height) / LANDMARK_IMAGE_SIZE
    face_tilt = (forehead[0] - chin[0]) / LANDMARK_IMAGE_SIZE

    return {
        "mouth_asymmetry": round(mouth_asymmetry, 3),
        "eye_asymmetry": round(eye_asymmetry, 3),
        "face_tilt": round(face_tilt, 3)
    }


def interpret_asymmetry(value, threshold=0.05):
    """Mengubah nilai asimetri menjadi keterangan kualitatif."""
    if value > threshold:
        return "signifikan"
    else:
        return "ringan"


def build_user_message(landmark_points, stroke_probability, is_stroke):
    """
    Membuat pesan penjelasan dalam bahasa Indonesia yang informatif dan mudah dipahami.
    Tanpa emoji dan menggunakan bahasa yang profesional.
    """
    if not landmark_points or len(landmark_points) < TOTAL_LANDMARKS:
        return "Maaf, tidak cukup data landmark wajah untuk analisis yang akurat. Pastikan foto wajah terlihat jelas."

    metrics = calculate_asymmetry_metrics(landmark_points)
    if not metrics:
        return "Tidak dapat menghitung metrik asimetri karena data landmark tidak lengkap."

    mouth_level = interpret_asymmetry(metrics["mouth_asymmetry"])
    eye_level = interpret_asymmetry(metrics["eye_asymmetry"])
    tilt_value = abs(metrics["face_tilt"])
    tilt_level = "miring" if tilt_value > 0.03 else "tegak"

    # Bagian hasil dan penjelasan
    if is_stroke:
        result_text = "Hasil: Terdeteksi indikasi STROKE"
        explanation = (
            f"Probabilitas: {stroke_probability*100:.1f}%.\n"
            "Model AI mendeteksi adanya asimetri pada wajah yang sering dikaitkan dengan stroke.\n"
            f"- Asimetri sudut mulut: {metrics['mouth_asymmetry']:.2f} ({mouth_level})\n"
            f"- Perbedaan tinggi kelopak mata: {metrics['eye_asymmetry']:.2f} ({eye_level})\n"
            f"- Kemiringan wajah: {tilt_value:.2f} ({tilt_level})\n\n"
            "Nilai-nilai ini menunjukkan adanya ketidakseimbangan otot wajah yang mungkin menjadi tanda awal stroke."
        )
        recommendation = (
            "Segera konsultasikan ke dokter atau tenaga medis untuk pemeriksaan lebih lanjut. "
            "Jangan menunda jika Anda juga mengalami gejala lain seperti:\n"
            "- Kesulitan berbicara atau memahami pembicaraan\n"
            "- Mati rasa atau kelemahan pada satu sisi tubuh\n"
            "- Sakit kepala hebat yang muncul tiba-tiba\n"
            "- Gangguan penglihatan pada satu atau kedua mata"
        )
    else:
        result_text = "Hasil: Normal (tidak terdeteksi stroke)"
        explanation = (
            f"Probabilitas stroke rendah: {stroke_probability*100:.1f}%.\n"
            "Wajah terdeteksi relatif simetris:\n"
            f"- Asimetri sudut mulut: {metrics['mouth_asymmetry']:.2f} ({mouth_level})\n"
            f"- Perbedaan tinggi kelopak mata: {metrics['eye_asymmetry']:.2f} ({eye_level})\n"
            f"- Kemiringan wajah: {tilt_value:.2f} ({tilt_level})\n\n"
            "Hasil ini menunjukkan tidak ada indikasi kuat stroke berdasarkan pola wajah."
        )
        recommendation = (
            "Hasil ini baik, namun tetaplah waspada. Jika Anda mengalami gejala seperti mati rasa, "
            "kesulitan bicara, atau sakit kepala hebat, segera periksakan ke dokter meskipun hasil ini normal."
        )

    disclaimer = (
        "\n\nPerhatian: Hasil ini hanya sebagai alat bantu awal dan bukan diagnosis medis. "
        "Keputusan medis harus selalu berdasarkan pemeriksaan oleh tenaga kesehatan profesional."
    )

    return result_text + "\n\n" + explanation + "\n\n" + recommendation + disclaimer


# ========================
# FASTAPI ENDPOINT
# ========================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Stroke Detection API",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Model tidak tersedia. Silakan coba lagi nanti.",
                "stroke": False
            }
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Gambar tidak valid. Mohon unggah file gambar yang benar.",
                    "stroke": False
                }
            )

        features, cropped, _, landmark_points = extract_landmarks_and_crop(image)
        if features is None:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "Tidak ada wajah yang terdeteksi. Pastikan foto menunjukkan wajah dengan jelas.",
                    "stroke": False
                }
            )

        # Prediksi
        input_tensor = torch.tensor([features], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)
            stroke_prob = probs[0][1].item()

        is_stroke = stroke_prob > 0.5
        confidence = stroke_prob if is_stroke else 1 - stroke_prob

        # Gambar landmark pada crop
        img_with_landmarks = None
        img_base64 = None
        if cropped is not None and landmark_points:
            img_with_landmarks = draw_landmarks_on_image(cropped, landmark_points)
            _, buffer = cv2.imencode('.jpg', img_with_landmarks)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Pesan untuk pengguna
        user_message = build_user_message(landmark_points, stroke_prob, is_stroke)

        return {
            "success": True,
            "stroke": is_stroke,
            "confidence": round(confidence, 4),
            "probability": round(stroke_prob, 4),
            "face_detected": True,
            "image_with_landmarks": img_base64,
            "result_label": "Stroke" if is_stroke else "Normal",
            "confidence_percentage": round(
                stroke_prob * 100 if is_stroke else (1 - stroke_prob) * 100, 1
            ),
            "message": user_message,
            # Untuk kompatibilitas
            "detailed_reason": user_message
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Terjadi kesalahan pada server. Silakan coba lagi.",
                "stroke": False
            }
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)