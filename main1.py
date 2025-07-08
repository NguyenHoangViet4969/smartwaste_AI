"""
Smart Waste AI – FastAPI + YOLOv8/YOLOv11
Version: 2025-07-09 (Improved)
"""

import os, json, logging, time, asyncio
from pathlib import Path
from uuid import uuid4
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import mimetypes

import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db as fdb

# ─────────────────────────────
# 1. CONFIGURATION & CONSTANTS
# ─────────────────────────────
ROOT = Path(__file__).parent.resolve()

# Model configuration
MODEL_PATH = ROOT / "weights" / os.getenv("MODEL_FILE", "best.pt")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))
DEVICE = os.getenv("DEVICE", "cpu")

# Directory setup
TMP_DIR, STATIC_DIR = ROOT / "temp", ROOT / "static"
TMP_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# File constraints
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", 2048))

# Cleanup configuration
TEMP_FILE_CLEANUP_HOURS = int(os.getenv("TEMP_FILE_CLEANUP_HOURS", 24))

# ─────────────────────────────
# 2. LOGGING SETUP
# ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────
# 3. LABELS & MAPPING
# ─────────────────────────────
def load_json_config(filename: str) -> Dict[str, Any]:
    """Load and validate JSON configuration files"""
    try:
        config_path = ROOT / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file {filename} not found")
        
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filename}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading {filename}: {e}")

CLASSES = load_json_config("classes.json")
MAPPING = load_json_config("mapping.json")
GROUP_TEXT = {"O": "Hữu cơ", "R": "Tái chế", "N": "Không tái chế"}

# ─────────────────────────────
# 4. FIREBASE SETUP
# ─────────────────────────────
class FirebaseManager:
    def __init__(self):
        self.ref_ai = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            cred_json = os.getenv("FIREBASE_CRED_JSON")
            if cred_json:
                cred = credentials.Certificate(json.loads(cred_json))
            else:
                cred_path = ROOT / "firebase_key.json"
                if not cred_path.exists():
                    raise FileNotFoundError(
                        f"Firebase credentials not found at {cred_path}. "
                        "Set FIREBASE_CRED_JSON environment variable or provide firebase_key.json"
                    )
                cred = credentials.Certificate(cred_path)

            firebase_admin.initialize_app(
                cred,
                {"databaseURL": "https://datn-5b6dc-default-rtdb.firebaseio.com/"}
            )
            self.ref_ai = fdb.reference("/waste/ai")
            logger.info("✅ Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Firebase initialization failed: {e}")
            raise

    async def push_data(self, info: Dict[str, Any]) -> bool:
        """Push data to Firebase asynchronously"""
        try:
            if not self.ref_ai:
                logger.error("Firebase not initialized")
                return False
            
            # Run Firebase update in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._update_firebase, info)
            return True
            
        except Exception as e:
            logger.error(f"Firebase push failed: {e}")
            return False
    
    def _update_firebase(self, info: Dict[str, Any]):
        """Synchronous Firebase update"""
        self.ref_ai.update({
            "group": info["code"],
            "label": info["label"],
            "confidence": info["conf"],
            "timestamp": time.time()
        })

# ─────────────────────────────
# 5. YOLO MODEL MANAGER
# ─────────────────────────────
class YOLOManager:
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load and initialize YOLO model"""
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
            self.model = YOLO(str(MODEL_PATH))
            self.model.fuse()
            self.model.to(DEVICE)
            logger.info(f"✅ YOLO model loaded on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"❌ YOLO model loading failed: {e}")
            raise
    
    def predict(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """Make prediction using YOLO model"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            results = self.model(img, conf=CONF_THRESH, verbose=False)[0]
            
            if not results.boxes:
                return None
            
            # Get the highest confidence detection
            box = results.boxes[0]
            cls_id = int(box.cls[0])
            
            if cls_id >= len(CLASSES):
                logger.warning(f"Invalid class ID: {cls_id}")
                return None
            
            label = CLASSES[cls_id]
            conf = float(box.conf[0])
            code = MAPPING.get(label, "N")  # Default to "N" if not found
            group = GROUP_TEXT.get(code, "Không xác định")
            
            return {
                "label": label,
                "conf": round(conf, 3),
                "code": code,
                "group": group,
                "bbox": box.xyxy[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

# ─────────────────────────────
# 6. IMAGE PROCESSING UTILITIES
# ─────────────────────────────
class ImageProcessor:
    @staticmethod
    def validate_image_file(file_path: Path) -> bool:
        """Validate image file format and size"""
        try:
            # Check file extension
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                return False
            
            # Check file size
            if file_path.stat().st_size > MAX_FILE_SIZE:
                return False
            
            # Try to read image
            img = cv2.imread(str(file_path))
            if img is None:
                return False
            
            # Check dimensions
            h, w = img.shape[:2]
            if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    @staticmethod
    def calculate_sharpness(img: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception as e:
            logger.error(f"Sharpness calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def draw_detection_box(img: np.ndarray, detection: Dict[str, Any], output_path: Path):
        """Draw detection box on image"""
        try:
            draw_img = img.copy()
            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{detection['label']} ({detection['conf']:.2f})"
            cv2.putText(draw_img, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imwrite(str(output_path), draw_img)
            
        except Exception as e:
            logger.error(f"Drawing detection box failed: {e}")

    @staticmethod
    def cleanup_old_files():
        """Clean up old temporary files"""
        try:
            cutoff_time = time.time() - (TEMP_FILE_CLEANUP_HOURS * 3600)
            
            for file_path in TMP_DIR.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Cleaned up old temp file: {file_path.name}")
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# ─────────────────────────────
# 7. APPLICATION LIFECYCLE
# ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("🚀 Starting Smart Waste AI application...")
    
    # Initialize components
    global firebase_manager, yolo_manager
    firebase_manager = FirebaseManager()
    yolo_manager = YOLOManager()
    
    # Initial cleanup
    ImageProcessor.cleanup_old_files()
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Smart Waste AI application...")
    
    # Cleanup resources
    try:
        firebase_admin.delete_app(firebase_admin.get_app())
    except:
        pass

# ─────────────────────────────
# 8. FASTAPI APPLICATION
# ─────────────────────────────
app = FastAPI(
    title="Smart Waste AI",
    description="AI-powered waste classification system",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global variables (initialized in lifespan)
firebase_manager: FirebaseManager = None
yolo_manager: YOLOManager = None

# ─────────────────────────────
# 9. API ENDPOINTS
# ─────────────────────────────
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "yolo_device": str(yolo_manager.model.device) if yolo_manager else "not_loaded",
        "timestamp": time.time()
    }

@app.get("/info")
async def get_info():
    """Get system information"""
    return {
        "classes": CLASSES,
        "mapping": MAPPING,
        "group_text": GROUP_TEXT,
        "config": {
            "conf_thresh": CONF_THRESH,
            "device": DEVICE,
            "max_file_size": MAX_FILE_SIZE,
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        }
    }

@app.post("/upload")
async def upload_image(request: Request):
    """Upload image for processing"""
    start_time = time.time()
    
    try:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Read request body
        body_start = time.time()
        data = await request.body()
        body_time = time.time() - body_start
        
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data received")
        
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Generate unique filename
        filename = f"{uuid4().hex}.jpg"
        file_path = TMP_DIR / filename
        
        # Save file
        save_start = time.time()
        file_path.write_bytes(data)
        save_time = time.time() - save_start
        
        # Validate image
        if not ImageProcessor.validate_image_file(file_path):
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        total_time = time.time() - start_time
        
        logger.info(f"✅ Image uploaded: {filename}")
        logger.info(f"⏱️ Upload times - Body: {body_time:.3f}s, Save: {save_time:.3f}s, Total: {total_time:.3f}s")
        
        return {
            "status": "uploaded",
            "filename": filename,
            "size": len(data),
            "times": {
                "body_read": round(body_time, 3),
                "file_save": round(save_time, 3),
                "total": round(total_time, 3)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get("/detect/{filename}")
async def detect_waste(filename: str, background_tasks: BackgroundTasks):
    """Detect waste in uploaded image"""
    start_time = time.time()
    
    try:
        # Find image file
        img_path = TMP_DIR / filename
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Load image
        load_start = time.time()
        img = cv2.imread(str(img_path))
        if img is None:
            img_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        load_time = time.time() - load_start
        
        # Calculate sharpness
        sharpness = ImageProcessor.calculate_sharpness(img)
        
        # Make prediction
        predict_start = time.time()
        detection = yolo_manager.predict(img)
        predict_time = time.time() - predict_start
        
        if detection is None:
            img_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="No waste object detected")
        
        # Save processed images
        save_start = time.time()
        cv2.imwrite(str(STATIC_DIR / "last.jpg"), img)
        
        # Draw detection box
        ImageProcessor.draw_detection_box(img, detection, STATIC_DIR / "last_boxed.jpg")
        save_time = time.time() - save_start
        
        # Push to Firebase (background task)
        firebase_start = time.time()
        firebase_success = await firebase_manager.push_data(detection)
        firebase_time = time.time() - firebase_start
        
        # Cleanup temp file
        cleanup_start = time.time()
        img_path.unlink(missing_ok=True)
        cleanup_time = time.time() - cleanup_start
        
        total_time = time.time() - start_time
        
        # Schedule background cleanup
        background_tasks.add_task(ImageProcessor.cleanup_old_files)
        
        logger.info(f"✅ Detection completed for {filename} in {total_time:.3f}s")
        
        return {
            "group": detection["group"],
            "label": detection["label"],
            "confidence": detection["conf"],
            "code": detection["code"],
            "sharpness": round(sharpness, 2),
            "image_url": "/static/last.jpg",
            "boxed_image_url": "/static/last_boxed.jpg",
            "firebase_success": firebase_success,
            "system_info": {
                "yolo_device": str(yolo_manager.model.device),
                "model_path": str(MODEL_PATH)
            },
            "performance": {
                "load_image": round(load_time, 3),
                "prediction": round(predict_time, 3),
                "save_images": round(save_time, 3),
                "firebase_push": round(firebase_time, 3),
                "cleanup": round(cleanup_time, 3),
                "total": round(total_time, 3)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed for {filename}: {e}")
        # Cleanup on error
        try:
            (TMP_DIR / filename).unlink(missing_ok=True)
        except:
            pass
        raise HTTPException(status_code=500, detail="Detection failed")

@app.post("/detect-direct")
async def detect_direct(request: Request, background_tasks: BackgroundTasks):
    """Direct detection without separate upload step"""
    try:
        # Upload image
        upload_result = await upload_image(request)
        filename = upload_result["filename"]
        
        # Detect waste
        detection_result = await detect_waste(filename, background_tasks)
        
        return {
            "upload": upload_result,
            "detection": detection_result
        }
        
    except Exception as e:
        logger.error(f"Direct detection failed: {e}")
        raise

@app.delete("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup of old files"""
    try:
        ImageProcessor.cleanup_old_files()
        return {"status": "cleanup_completed"}
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

# ─────────────────────────────
# 10. ERROR HANDLERS
# ─────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

# ─────────────────────────────
# 11. STARTUP MESSAGE
# ─────────────────────────────
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Smart Waste AI server...")
    logger.info(f"📁 Model path: {MODEL_PATH}")
    logger.info(f"🎯 Confidence threshold: {CONF_THRESH}")
    logger.info(f"💻 Device: {DEVICE}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)