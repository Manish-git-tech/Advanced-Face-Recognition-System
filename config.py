# File Structure:
# ├── config.py
# ├── face_processor.py
# ├── database_handler.py
# ├── employee_registrar.py
# ├── recognition_app.py
# └── requirements.txt

# config.py
import os

CONFIG = {
    "FACE_MODEL_NAME": "buffalo_l",
    "DETECTION_THRESHOLD": 0.6,
    "DATABASE_NAME": "employees.db",
    "EMPLOYEE_DATA_ROOT": "data/employees",
    "EMBEDDINGS_PATH": "employee_embeddings",
    "MAX_CAPTURE_IMAGES": 10,
    "FACE_DETECTION_CONFIDENCE": 0.6
}

# Create necessary directories
os.makedirs(CONFIG["EMPLOYEE_DATA_ROOT"], exist_ok=True)
os.makedirs(CONFIG["EMBEDDINGS_PATH"], exist_ok=True)
