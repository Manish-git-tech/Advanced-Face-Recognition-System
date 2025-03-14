# recognition_app.py
import cv2
import numpy as np
from config import CONFIG
from face_processor import FaceProcessor
from database_handler import DatabaseManager
import time
import winsound
from datetime import datetime, timedelta
from PIL import Image


def play_success():
    """Play a success sound (high frequency beep)."""
    frequency = 1000  # Frequency in Hertz
    duration = 500    # Duration in milliseconds (500 ms = 0.5 seconds)
    winsound.Beep(frequency, duration)

class RecognitionApp:
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.db = DatabaseManager()
        self._load_known_embeddings()  # Load employees
        self.current_users = set()     # Employee IDs currently seen
        self.current_strangers = {}    # Dictionary to track unknown faces, keyed by a temporary ID
        self.last_log_times = {}       # For employee log cooldown
        self.stranger_log_cooldown = timedelta(seconds=10)  # adjust if needed
        self.next_stranger_id = 1      # Used to assign a temporary ID to new unknown faces

    def _load_known_embeddings(self):
        self.known_embeddings = {
            employee['employee_institute_id']: {
                **employee,
                'original_encoding': employee['encoding'].copy(),
                'embedding_history': [employee['encoding']]
            } 
            for employee in self.db.get_employee_data()
        }


    def _load_known_embeddings(self):
        self.known_embeddings = {
            employee['employee_institute_id']: {
                **employee,
                'original_encoding': employee['encoding'].copy(),
                'embedding_history': [employee['encoding']]
            } 
            for employee in self.db.get_employee_data()
        }
    def _get_or_create_stranger(self, face, frame):
        """Check if the unknown face matches an already tracked stranger.
           If yes, update that record; otherwise, create a new stranger record.
        """
        # Extract bounding box and compute its center coordinates
        x1, y1, x2, y2 = map(int, face.bbox)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Try to find an existing stranger with a nearby bounding box center.
        matching_stranger = None
        for rec in self.current_strangers.values():
            bx1, by1, bx2, by2 = map(int, rec['bbox'])
            rec_center_x = (bx1 + bx2) / 2
            rec_center_y = (by1 + by2) / 2
            dist = ((center_x - rec_center_x)**2 + (center_y - rec_center_y)**2)**0.5
            if dist < 50:  # Using 50 pixels as a threshold; adjust as needed.
                matching_stranger = rec
                break

        if matching_stranger:
            # Update the existing record.
            matching_stranger['bbox'] = face.bbox
            matching_stranger['last_seen'] = datetime.now()
            return matching_stranger
        else:
            # New stranger—create a new record.
            stranger_id = f"stranger_{int(time.time())}_{self.next_stranger_id}"
            self.next_stranger_id += 1

            # Extract face image from frame.
            face_img = frame[y1:y2, x1:x2]
            if face_img is None or face_img.size == 0:
                print("Warning: Empty face image extracted. Skipping this face.")
                return None
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_pil = Image.fromarray(face_img_rgb)
            
            # Log stranger entry event in the database.
            self.db.log_stranger_entry(face_img_pil)
            # Create and store the new stranger record.
            new_stranger = {
                "temp_id": stranger_id,
                "bbox": face.bbox,
                "face": face_img_pil,
                "last_seen": datetime.now()
            }
            self.current_strangers[stranger_id] = new_stranger
            print(f"Stranger entry logged for {stranger_id}")
            return new_stranger

    def process_faces(self, frame):
        faces = self.face_processor.detect_faces(frame)
        recognized_employees = []
        recognized_strangers = []

        for face in faces:
            current_embedding = face.embedding

            # Employee matching block:
            best_employee = max(
                self.known_embeddings.values(),
                key=lambda emp: self.face_processor.calculate_similarity(current_embedding, emp['encoding']),
                default=None
            )
            if best_employee:
                similarity = self.face_processor.calculate_similarity(current_embedding, best_employee['encoding'])
                if similarity > CONFIG["DETECTION_THRESHOLD"]:
                    best_employee['confidence'] = similarity
                    best_employee['bbox'] = face.bbox
                    best_employee['current_embedding'] = current_embedding
                    # Optionally update embedding
                    updated_embedding = self.face_processor.update_embedding(best_employee)
                    best_employee['encoding'] = updated_embedding
                    self.db.update_employee_embedding(best_employee['employee_institute_id'], updated_embedding)
                    recognized_employees.append(best_employee)
                    continue  # Found an employee match, no stranger logging needed.

            # Stranger processing:
            stranger_record = self._get_or_create_stranger(face, frame)
            if stranger_record is not None:
                recognized_strangers.append(stranger_record)

        return recognized_employees, recognized_strangers
    def determine_log_type(self, employee_id):
        last_entry = self.db.get_last_entry(employee_id)
        last_exit = self.db.get_last_exit(employee_id)

        if not last_entry and not last_exit:
            return "entry"  # Default to entry for new employees
        return "exit" if last_entry and (not last_exit or last_entry > last_exit) else "entry"

    def log_access(self, employee_id, employee_name, log_type):
        current_time = datetime.now()
        last_log_time = self.last_log_times.get(employee_id)

        if last_log_time and (current_time - last_log_time) < self.log_cooldown:
            print(f"Skipped logging for {employee_name} (last log was less than 1 minute ago)")
            return

        if log_type == "entry":
            self.db.log_entry(employee_id, employee_name)
        else:
            self.db.log_exit(employee_id, employee_name)

        self.last_log_times[employee_id] = current_time
        print(f"{log_type.capitalize()} logged for {employee_name}")
        play_success()
    

    def log_stranger_access(self, stranger_id, stranger_number, log_type):
        current_time = datetime.now()
        last_time = self.last_stranger_log_times.get(stranger_id)
        if last_time and (current_time - last_time) < self.log_cooldown:
            print(f"Skipped logging for {stranger_number} (cooldown active)")
            return
        if log_type == "entry":
            self.db.log_stranger_entry(stranger_id, current_time)
        else:
            self.db.log_stranger_exit(stranger_id, current_time)
        self.last_stranger_log_times[stranger_id] = current_time
        print(f"Stranger {stranger_number} {log_type} logged at {current_time}")
        play_success()
        
    def determine_log_type_stranger(self, stranger_id):
        last_entry = self.db.get_last_stranger_entry(stranger_id)  # Should return a timestamp or None
        last_exit = self.db.get_last_stranger_exit(stranger_id)
        if not last_entry and not last_exit:
            return "entry"  # Default when no logs exist
        return "exit" if last_entry and (not last_exit or last_entry > last_exit) else "entry"
    
    def display_employee_info(self, frame, employee):
        bbox = employee['bbox'].astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{employee['name']}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {employee['confidence']:.2f}",
                    (bbox[0], bbox[3] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def display_stranger_info(self, frame, stranger):
        bbox = list(map(int, stranger['bbox']))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(frame, "Stranger", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            recognized_employees, recognized_strangers = self.process_faces(frame)

            # Process employee logging and display.
            for employee in recognized_employees:
                self.display_employee_info(frame, employee)
                if employee['id'] not in self.current_users:
                    self.current_users.add(employee['id'])
                    log_type = self.determine_log_type(employee['id'])
                    self.log_access(employee['id'], employee['name'], log_type)

            # Process stranger logging and display.
            current_tracked_strangers = set()
            for stranger in recognized_strangers:
                self.display_stranger_info(frame, stranger)
                current_tracked_strangers.add(stranger['temp_id'])
                # The _get_or_create_stranger function already logs an entry if needed.
                # Update last_seen (already done in _get_or_create_stranger)

            # For any strangers previously tracked but not detected now, log exit events.
            for temp_id in list(self.current_strangers.keys()):
                if temp_id not in current_tracked_strangers:
                    record = self.current_strangers.pop(temp_id)
                    self.db.log_stranger_exit(record['face'])
                    print(f"Stranger exit logged for {temp_id}")

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
# Usage
if __name__ == "__main__":
    app = RecognitionApp()
    app.run()