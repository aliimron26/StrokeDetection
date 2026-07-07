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


def get_risk_category(stroke_probability):
    if stroke_probability >= 0.7:
        return "tinggi"
    elif stroke_probability >= 0.3:
        return "sedang"
    else:
        return "rendah"


def build_user_message(landmark_points, stroke_probability, is_stroke):
    if not landmark_points or len(landmark_points) < TOTAL_LANDMARKS:
        return "Maaf, data wajah tidak cukup. Pastikan foto wajah terlihat jelas dan menghadap kamera."

    risk_category = get_risk_category(stroke_probability)
    percentage = stroke_probability * 100 if is_stroke else (1 - stroke_probability) * 100

    how_it_works = (
        "AI menganalisis 478 titik di seluruh wajah untuk mendeteksi ketidaksimetrisan halus "
        "yang sering terkait dengan stroke atau TIA, seperti kelemahan otot wajah, sudut mulut tidak sejajar, "
        "atau perbedaan bukaan mata. Karena AI menilai semua titik secara bersamaan, hasil akhirnya adalah "
        "satu skor keyakinan menyeluruh."
    )

    if is_stroke:
        result_text = f"Hasil: Terdeteksi indikasi STROKE (keyakinan {percentage:.1f}%)"
        explanation = (
            f"Tingkat risiko: {risk_category}. Pola ketidaksimetrisan wajah yang ditemukan cukup kuat, "
            "sehingga AI mengklasifikasikan gambar ini sebagai indikasi stroke. Ini bukan diagnosis pasti, "
            "melainkan pola yang mirip dengan kasus stroke yang pernah dipelajari AI."
        )
        recommendation = (
            "Langkah selanjutnya: Segera konsultasikan ke dokter atau tenaga medis. "
            "Jangan menunda jika juga mengalami gejala lain:\n"
            "- Kesulitan bicara atau memahami pembicaraan\n"
            "- Mati rasa atau kelemahan pada satu sisi tubuh\n"
            "- Sakit kepala hebat yang muncul tiba-tiba tanpa sebab jelas\n"
            "- Gangguan penglihatan pada satu atau kedua mata \n"
            "- Pusing berat atau kehilangan keseimbangan/koordinasi\n"
            "- Kebingungan mendadak atau kesulitan memahami orang lain"
        )
    else:
        result_text = f"Hasil: Normal (tidak terdeteksi stroke) - keyakinan {percentage:.1f}%"
        explanation = (
            f"Tingkat risiko stroke: {risk_category}. AI tidak menemukan pola ketidaksimetrisan yang cukup kuat "
            "untuk dikategorikan sebagai indikasi stroke. Wajah Anda relatif simetris menurut pola yang dipelajari AI."
        )
        recommendation = (
            "Meskipun hasil ini baik, tetaplah waspada. Jika Anda mengalami salah satu gejala berikut, "
            "segera periksakan ke dokter:\n"
            "- Kesulitan bicara atau memahami pembicaraan\n"
            "- Mati rasa atau kelemahan pada satu sisi tubuh\n"
            "- Sakit kepala hebat yang muncul tiba-tiba\n"
            "- Gangguan penglihatan mendadak\n"
            "- Pusing atau kehilangan keseimbangan\n"
            "- Kebingungan mendadak"
        )

    disclaimer = (
        ""
    )

    return f"{result_text}\n\n{explanation}\n\n{recommendation}{disclaimer}"

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