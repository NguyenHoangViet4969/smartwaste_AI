# ================= 1. import / cấu hình =================
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np, cv2, json, pathlib, time, logging
from datetime import datetime

# import firebase_admin                                 # ★ new
# from firebase_admin import credentials, db           # ★

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

BASE   = pathlib.Path(__file__).parent
CLS    = json.load(open(BASE / "classes.json"))
GROUP  = json.load(open(BASE / "mapping.json"))

# ---------- YOLO ----------
model  = YOLO(BASE / "weights" / "best.pt")
model.fuse()

# # ---------- Firebase ----------
# cred = credentials.Certificate(BASE / "serviceAccountKey.json")  # tải từ console
# firebase_admin.initialize_app(cred, {
#     "databaseURL": "https://datn-5b6dc-default-rtdb.firebaseio.com/"
# })

# ---------- Webcam ----------
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    raise RuntimeError("❌ Không mở được webcam")

# ---------- FastAPI ----------
app = FastAPI(title="SmartWaste-Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Thư mục debug ----------
DEBUG_DIR = BASE / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

# ================= 2. endpoint /detect (giữ nguyên) =================
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

    vis = r.plot()
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    cv2.imwrite(str(DEBUG_DIR / f"{ts}_{file.filename}"), vis)

    logging.info("Detect %-20s %5.1f ms  saved→ %s",
                 file.filename, infer_ms, f"{ts}_{file.filename}")

    return {"detections": dets}

# ================= 3. endpoint /snapshot (mới, cho ESP) ==============
# @app.get("/snapshot")
# def snapshot(conf: float = 0.25):
#     ok, frame = cam.read()
#     if not ok:
#         return {"error": "camera"}

#     t0 = time.perf_counter()
#     r  = model(frame, conf=conf, verbose=False)[0]
#     ms = (time.perf_counter() - t0) * 1000

#     if not len(r.boxes):
#         grp, lbl, sc = "N", "unknown", 0
#     else:
#         idx = int(r.boxes.conf.argmax())
#         lbl = CLS[int(r.boxes.cls[idx])]
#         grp = GROUP.get(lbl, "N")
#         sc  = float(r.boxes.conf[idx])

#     # 🔥 ghi kết quả lên Firebase
#     db.reference("/smartwaste/current").set(grp)
#     # (nếu ESP đặt trigger) → reset
#     db.reference("/smartwaste/trigger").set(False)

#     # Lưu ảnh debug
#     vis = r.plot()
#     ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
#     cv2.imwrite(str(DEBUG_DIR / f"{ts}_snapshot.jpg"), vis)

#     logging.info("Snapshot %-12s %4.1f ms  ⇒ %s", lbl, ms, grp)
#     return {"group": grp, "label": lbl, "conf": round(sc,3), "time_ms": round(ms,1)}
