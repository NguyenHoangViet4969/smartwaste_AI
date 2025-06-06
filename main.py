# ================= 1. import / cấu hình =================
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np, cv2, json, pathlib, time, logging
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

BASE   = pathlib.Path(__file__).parent
CLS    = json.load(open(BASE / "classes.json"))
GROUP  = json.load(open(BASE / "mapping.json"))

# ---------- YOLO ----------
model  = YOLO(BASE / "weights" / "best.pt")
model.fuse()

# ---------- Firebase ----------
cred = credentials.Certificate(BASE / "serviceAccountKey.json")  # 🔐
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://datn-5b6dc-default-rtdb.firebaseio.com/"
})

# ---------- Webcam ----------
try:
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = None
        logging.warning("⚠️ Không mở được webcam (chưa cắm hoặc không hỗ trợ)")
except:
    cam = None
    logging.warning("⚠️ Không thể khởi tạo webcam (lỗi thư viện hoặc thiết bị)")

# ---------- FastAPI ----------
app = FastAPI(
    title="SmartWaste-Detector",
    docs_url="/docs",
    redoc_url=None
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Thư mục debug ----------
DEBUG_DIR = BASE / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

# ================= 2. endpoint /detect =================
@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8),
                       cv2.IMREAD_COLOR)

    t0 = time.perf_counter()
    r  = model(img, conf=conf, verbose=False)[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    dets = []
    for box, score, cls_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        lbl = CLS[int(cls_id)]
        dets.append({
            "label": lbl,
            "group": GROUP.get(lbl, "N"),
            "conf" : round(float(score), 4),
            "box"  : [round(x, 2) for x in box.cpu().tolist()]
        })

    # 🔥 Gửi group lên Firebase nếu có
    if dets:
        best = max(dets, key=lambda d: d['conf'])
        group = best['group']
        db.reference("/smartwaste/current").set(group)
        logging.info("🔥 Gửi group lên Firebase: %s", group)

    vis = r.plot()
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    cv2.imwrite(str(DEBUG_DIR / f"{ts}_{file.filename}"), vis)

    logging.info("Detect %-20s %5.1f ms  saved→ %s",
                 file.filename, infer_ms, f"{ts}_{file.filename}")

    return {"detections": dets}

# ================= 3. endpoint /snapshot (cho ESP) =================
@app.get("/snapshot")
def snapshot(conf: float = 0.25):
    if cam is None:
        return {"error": "camera_not_available"}

    ok, frame = cam.read()
    if not ok:
        return {"error": "camera_read_failed"}

    t0 = time.perf_counter()
    r  = model(frame, conf=conf, verbose=False)[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    if not len(r.boxes):
        grp, lbl, conf_score = "N", "unknown", 0
    else:
        idx = int(r.boxes.conf.argmax())
        lbl = CLS[int(r.boxes.cls[idx])]
        grp = GROUP.get(lbl, "N")
        conf_score = float(r.boxes.conf[idx])

    # 🔥 Gửi kết quả lên Firebase
    db.reference("/smartwaste/current").set(grp)
    db.reference("/smartwaste/trigger").set(False)

    vis = r.plot()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv2.imwrite(str(DEBUG_DIR / f"{ts}_snapshot.jpg"), vis)

    logging.info("Snapshot %-12s %4.1f ms  ⇒ %s", lbl, infer_ms, grp)
    return {"group": grp, "label": lbl, "conf": round(conf_score,3), "time_ms": round(infer_ms,1)}

# ================= Root =================
@app.get("/")
def root():
    return {"message": "SmartWaste AI server is running"}
