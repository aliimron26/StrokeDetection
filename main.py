import os
import sys
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
from io import BytesIO
from PIL import Image
warnings.filterwarnings('ignore')

# Import MediaPipe
from mediapipe.python.solutions import face_detection as mp_face_detection
from mediapipe.python.solutions import face_mesh as mp_face_mesh

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Inisialisasi MediaPipe
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ---------- Model ----------
class StrokeDetectionMLP(nn.Module):
    def __init__(self, input_dim=956, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hdim), nn.BatchNorm1d(hdim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model_path = os.getenv("MODEL_PATH", "best_model.pth")
model = None
if os.path.exists(model_path):
    try:
        model = StrokeDetectionMLP()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print(f"Model file not found at {model_path}")

# ---------- Helper: Ekstraksi landmark dan crop ----------
def extract_landmarks_and_crop(image):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    det = face_detection.process(rgb)
    if not det.detections:
        return None, None, None, False

    bbox = det.detections[0].location_data.relative_bounding_box
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
        return None, None, None, False

    crop_resized = cv2.resize(cropped, (300, 300))
    rgb_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    mesh = face_mesh.process(rgb_crop)
    if not mesh.multi_face_landmarks:
        return None, None, None, False

    landmarks = []
    for lm in mesh.multi_face_landmarks[0].landmark:
        landmarks.append(lm.x * 300)
        landmarks.append(lm.y * 300)

    features = np.array(landmarks, dtype=np.float32)
    if features.std() > 1e-6:
        features = (features - features.mean()) / (features.std() + 1e-6)

    # Simpan landmarks asli (dalam koordinat 300x300) untuk digambar
    landmark_points = [(lm.x * 300, lm.y * 300) for lm in mesh.multi_face_landmarks[0].landmark]

    return features.tolist(), cropped, (x, y, box_w, box_h), landmark_points

# ---------- Helper: Gambar landmark pada citra ----------
def draw_landmarks_on_image(image_bgr, landmark_points, size=(300,300)):
    """
    Menggambar titik-titik landmark pada gambar (BGR).
    Mengembalikan gambar yang sudah digambar (BGR).
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]
    # Jika ukuran gambar bukan 300x300, skala landmark
    scale_x = w / size[0]
    scale_y = h / size[1]
    for (lx, ly) in landmark_points:
        x = int(lx * scale_x)
        y = int(ly * scale_y)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # titik hijau
    return img

# ---------- Helper: Generate detailed reason ----------
def generate_detailed_reason(landmark_points, probability, is_stroke):
    """
    Membuat penjelasan berdasarkan beberapa metrik asimetri sederhana.
    """
    if not landmark_points or len(landmark_points) < 468:
        return "Tidak cukup data landmark untuk analisis mendalam."

    left_mouth = landmark_points[61]   # (x,y)
    right_mouth = landmark_points[291]
    left_eye_top = landmark_points[33]
    left_eye_bottom = landmark_points[133]
    right_eye_top = landmark_points[362]
    right_eye_bottom = landmark_points[263]
    chin = landmark_points[152]
    forehead = landmark_points[1]

    # Hitung asimetri mulut (horizontal)
    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    mouth_asymmetry = abs(left_mouth[0] - right_mouth[0]) / 300.0  # normalisasi

    # Asimetri kelopak mata (jarak vertikal)
    left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
    right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
    eye_asymmetry = abs(left_eye_height - right_eye_height) / 300.0

    # Kemiringan wajah (antara dahi dan dagu)
    face_tilt = (forehead[0] - chin[0]) / 300.0

    # Buat narasi
    reason = "Analisis asimetri wajah dari 468 titik landmark menunjukkan:\n"
    reason += f"- Asimetri sudut mulut: {mouth_asymmetry:.3f} (semakin tinggi indikasi asimetri).\n"
    reason += f"- Perbedaan tinggi kelopak mata: {eye_asymmetry:.3f}.\n"
    reason += f"- Kemiringan wajah: {face_tilt:.3f}.\n\n"

    if is_stroke:
        reason += "Berdasarkan model AI, probabilitas stroke terdeteksi sebesar "
        reason += f"{probability*100:.1f}%. Ini menunjukkan adanya indikasi kuat terhadap "
        reason += "kelainan saraf yang sering terkait dengan stroke. Segera konsultasikan ke dokter."
    else:
        reason += "Probabilitas stroke rendah ({:.1f}%). Hasil ini menunjukkan wajah relatif simetris, "
        reason += "namun tetap waspada jika ada gejala lain."
        reason = reason.format(probability*100)

    # Tambahkan saran
    reason += "\n\n⚠️ Hasil ini hanya sebagai alat bantu awal, bukan diagnosis medis."
    return reason

# ---------- Endpoint ----------
@app.get("/")
async def root():
    return {"message": "Stroke Detection API", "model_loaded": model is not None}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={
            "success": False,
            "error": "Model not loaded",
            "stroke": False
        })

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "Invalid image",
                "stroke": False
            })

        # Ekstrak fitur, crop, dan landmark points
        features, cropped, bbox, landmark_points = extract_landmarks_and_crop(image)
        if features is None:
            return JSONResponse(status_code=200, content={
                "success": False,
                "error": "No face detected",
                "stroke": False
            })

        # Prediksi
        input_tensor = torch.tensor([features], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)
            stroke_prob = probs[0][1].item()

        is_stroke = stroke_prob > 0.5
        confidence = stroke_prob if is_stroke else 1 - stroke_prob

        # Gambar landmark pada crop
        if cropped is not None and landmark_points:
            img_with_landmarks = draw_landmarks_on_image(cropped, landmark_points)
            # Encode ke base64
            _, buffer = cv2.imencode('.jpg', img_with_landmarks)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            img_base64 = None

        # Buat detailed reason
        detailed_reason = generate_detailed_reason(landmark_points, stroke_prob, is_stroke)

        return {
            "success": True,
            "stroke": is_stroke,
            "confidence": round(confidence, 4),
            "probability": round(stroke_prob, 4),
            "face_detected": True,
            "image_with_landmarks": img_base64,
            "detailed_reason": detailed_reason
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e),
            "stroke": False
        })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)