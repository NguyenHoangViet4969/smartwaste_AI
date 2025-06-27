"""
Smart Waste AI – FastAPI + YOLOv8
Version: 2025-06-27
"""

import os, json, logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db as fdb

# ────────────────────────────────────────────────────────────
# 1. THƯ MỤC & CẤU HÌNH CHUNG
# ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()

MODEL_PATH = ROOT / "weights" / os.getenv("MODEL_FILE", "best.pt")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))

TMP_DIR, STATIC_DIR = ROOT / "temp", ROOT / "static"
TMP_DIR.mkdir(exist_ok=True), STATIC_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────
# 2. LABELS & MAPPING
# ────────────────────────────────────────────────────────────
CLASSES = json.load(open(ROOT / "classes.json", encoding="utf-8"))
MAPPING = json.load(open(ROOT / "mapping.json", encoding="utf-8"))
GROUP_TEXT = {"O": "Hữu cơ", "R": "Tái chế", "N": "Không tái chế"}

# ────────────────────────────────────────────────────────────
# 3. KẾT NỐI FIREBASE (Env trước, file sau)
# ────────────────────────────────────────────────────────────
cred_json = os.getenv("FIREBASE_CRED_JSON")
if cred_json:                                        # production (Render)
    cred = credentials.Certificate(json.loads(cred_json))
else:                                                # local
    cred_path = ROOT / "firebase_key.json"
    if not cred_path.exists():
        raise FileNotFoundError(
            f"Không thấy {cred_path}. Hãy copy service-account JSON vào dự án "
            "hoặc đặt biến môi trường FIREBASE_CRED_JSON."
        )
    cred = credentials.Certificate(cred_path)

firebase_admin.initialize_app(
    cred,
    {"databaseURL": "https://datn-5b6dc-default-rtdb.firebaseio.com/"},
)
ref_ai = fdb.reference("/waste/ai")

# ────────────────────────────────────────────────────────────
# 4. KHỞI TẠO YOLOv8
# ────────────────────────────────────────────────────────────
model = YOLO(str(MODEL_PATH))
model.fuse()

# ────────────────────────────────────────────────────────────
# 5. FASTAPI APP
# ────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Waste AI")
log = logging.getLogger("uvicorn.error")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ────────────────────────────────────────────────────────────
# 6. HÀM TIỆN ÍCH
# ────────────────────────────────────────────────────────────
def sharpness_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def classify(img: np.ndarray, box_path: Path | None = None) -> dict | None:
    res = model(img, conf=CONF_THRESH, verbose=False)[0]
    if not res.boxes:
        return None

    box   = res.boxes[0]
    label = CLASSES[int(box.cls[0])]
    conf  = float(box.conf[0])
    code  = MAPPING.get(label, "N")
    group = GROUP_TEXT[code]

    if box_path:
        draw = img.copy()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(draw, f"{label} ({conf:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(box_path), draw)

    return {"label": label, "conf": round(conf, 3), "code": code, "group": group}

def push_to_firebase(info: dict):
    ref_ai.update({"group": info["group"]})

# ────────────────────────────────────────────────────────────
# 7. ENDPOINTS
# ────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_img(req: Request):
    data = await req.body()
    fname = TMP_DIR / f"{uuid4().hex}.jpg"
    fname.write_bytes(data)
    return {"status": "uploaded", "file": fname.name}

@app.get("/detect")
def detect_best():
    imgs = list(TMP_DIR.glob("*.jpg"))
    if not imgs:
        return {"error": "no_images"}

    best_path  = max(imgs, key=lambda p: sharpness_score(cv2.imread(str(p))))
    best_img   = cv2.imread(str(best_path))
    best_score = sharpness_score(best_img)
    log.info(f"Best image = {best_path.name} ({best_score:.2f})")

    info = classify(best_img, box_path=STATIC_DIR / "last_boxed.jpg")
    if not info:
        return {"error": "no_object"}

    cv2.imwrite(str(STATIC_DIR / "last.jpg"), best_img)
    push_to_firebase(info)

    for p in imgs:
        p.unlink(missing_ok=True)

    return {
        "group": info["group"],
        "label": info["label"],
        "conf": info["conf"],
        "sharpness": round(best_score, 2),
        "image_with_box": "/static/last_boxed.jpg",
    }
