"""
FastAPI YOLO + Firebase trigger
  • POST /detect   : upload image
  • GET  /snapshot : pull image from ESP32-CAM
    → Sau mỗi infer ➜ /waste/ai = true và /waste/group = loại rác
"""
import os, time, cv2, json, logging, base64
from datetime import datetime
from pathlib import Path
from io import StringIO

import numpy as np, httpx
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, db as fdb


# ─── Config ───────────────────────────────────────────
ROOT        = Path(__file__).parent
MODEL_PATH  = ROOT / "weights" / "best.pt"
ESP_CAM_URL = "http://192.168.118.187/capture"
DEBUG_DIR   = ROOT / "snapshots"; DEBUG_DIR.mkdir(exist_ok=True)
CONF_DEF    = 0.25

# ─── Class & Mapping ─────────────────────────────────
CLASSES    = json.loads((ROOT / "classes.json").read_text())
MAPPING    = json.loads((ROOT / "mapping.json").read_text())  # name -> R/O/N
GROUP_TXT  = {"R": "Tái chế", "O": "Hữu cơ", "N": "Không tái chế"}

# ─── Firebase Init ───────────────────────────────────
firebase_json_str = os.getenv("FIREBASE_KEY_JSON")
firebase_dict = json.load(StringIO(firebase_json_str))

cred = credentials.Certificate(firebase_dict)
firebase_admin.initialize_app(
    cred, {"databaseURL": "https://datn-5b6dc-default-rtdb.firebaseio.com/"}
)


def push_trigger(info: dict) -> None:
    fdb.reference("/waste").update({
        "ai": True,
        "group": info["group"]
    })

# ─── Load YOLO model ─────────────────────────────────
model = YOLO(str(MODEL_PATH))
if bool(int(os.getenv("FUSE", 1))):
    model.fuse()

# ─── FastAPI App ─────────────────────────────────────
app = FastAPI(title="SmartWaste AI")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
log = logging.getLogger("uvicorn.error")

# ─── Helpers ─────────────────────────────────────────
def save_dbg(img: np.ndarray, tag: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(str(DEBUG_DIR / f"{ts}_{tag}.jpg"), img)

def infer(img: np.ndarray, conf: float = CONF_DEF):
    res = model(img, conf=conf, verbose=False)[0]
    if not len(res.boxes):
        return res.plot(), None

    b       = res.boxes[0]
    idx     = int(b.cls[0])
    label   = CLASSES[idx]
    code    = MAPPING.get(label, "N")
    group   = GROUP_TXT[code]
    score   = float(b.conf[0])
    info = {"label": label, "code": code, "group": group, "conf": score}
    return res.plot(), info

# ─── Endpoints ───────────────────────────────────────
@app.post("/detect")
async def detect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float = CONF_DEF,
):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    vis, info = await run_in_threadpool(infer, img, conf)
    if info is None:
        return {"error": "no_object"}

    _, buffer = cv2.imencode(".jpg", vis)
    img_b64 = base64.b64encode(buffer).decode()

    # background_tasks.add_task(save_dbg, vis, "upload")
    background_tasks.add_task(push_trigger, info)

    return {**info, "image": img_b64}

@app.get("/snapshot")
async def snapshot(
    background_tasks: BackgroundTasks,
    conf: float = CONF_DEF,
):
    print("📸 snapshot endpoint triggered")
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(ESP_CAM_URL)
            r.raise_for_status()
    except httpx.RequestError as e:
        log.error("🚫 ESP32-CAM unreachable: %s", e)
        return {"error": "esp_cam_unreachable"}

    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    vis, info = await run_in_threadpool(infer, img, conf)
    if info is None:
        return {"error": "no_object"}

    _, buffer = cv2.imencode(".jpg", vis)
    img_b64 = base64.b64encode(buffer).decode()

    # background_tasks.add_task(save_dbg, vis, "snapshot")
    background_tasks.add_task(push_trigger, info)

    return {**info, "image": img_b64}
