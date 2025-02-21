# recognition_app.py
import cv2
import numpy as np
from config import CONFIG
from face_processor import FaceProcessor
from database_handler import DatabaseManager
import time
import winsound
from datetime import datetime, timedelta

def play_success():
    """Play a success sound (high frequency beep)."""
    frequency = 1000  # Frequency in Hertz
    duration = 500    # Duration in milliseconds (500 ms = 0.5 seconds)
    winsound.Beep(frequency, duration)

class RecognitionApp:
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.db = DatabaseManager()
        self._load_known_embeddings()
        self.current_users = set()
        self.last_log_times = {}
        self.confidence_threshold = 0.6  # Threshold for resetting embedding
        self.log_cooldown = timedelta(minutes=1)  # 1 minute cooldown

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

    def recognize_employees(self, frame):
        faces = self.face_processor.detect_faces(frame)
        recognized_employees = []

        for face in faces:
            current_embedding = face.embedding
            for i in range(1):
                best_match = max(
                    self.known_embeddings.values(),
                    key=lambda emp: self.face_processor.calculate_similarity(current_embedding, emp['encoding']),
                    default=None
                )

            if best_match:
                similarity = self.face_processor.calculate_similarity(current_embedding, best_match['encoding'])
                best_match['confidence'] = similarity
                best_match['bbox'] = face.bbox
                best_match['current_embedding'] = current_embedding

                if similarity > CONFIG["DETECTION_THRESHOLD"]:
                    updated_embedding = self.face_processor.update_embedding(best_match)
                    best_match['encoding'] = updated_embedding
                    self.db.update_employee_embedding(best_match['employee_institute_id'], updated_embedding)
                    recognized_employees.append(best_match)

        return recognized_employees

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

    def display_employee_info(self, frame, employee):
        bbox = employee['bbox'].astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{employee['name']}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {employee['confidence']:.2f}", (bbox[0], bbox[3] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            recognized_employees = self.recognize_employees(frame)

            for employee in recognized_employees:
                self.display_employee_info(frame, employee)
                if employee['id'] not in self.current_users:
                    self.current_users.add(employee['id'])
                    log_type = self.determine_log_type(employee['id'])
                    self.log_access(employee['id'], employee['name'], log_type)

            # Remove users who are no longer in the frame
            current_ids = set(emp['id'] for emp in recognized_employees)
            self.current_users = self.current_users.intersection(current_ids)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    app = RecognitionApp()
    app.run()