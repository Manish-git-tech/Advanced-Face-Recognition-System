import cv2
import os
from PIL import Image
import numpy as np
from database_handler import DatabaseManager
from config import CONFIG
from face_processor import FaceProcessor
import time

class EmployeeRegistrar:
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.db = DatabaseManager()
    
    def capture_face_samples(self,name1=None,institute_id=None):
        """Interactive face registration through webcam"""
        if not name1:
            employee_name = input("Enter employee name: ").strip()
        else:
            employee_name = name1
        if not institute_id:
            employee_institute_id = input("Enter employee_institute_id: ").strip()
        else:
            employee_institute_id = institute_id
        employee_data = f"{employee_name}_{employee_institute_id}"
        save_path = os.path.join(CONFIG["EMPLOYEE_DATA_ROOT"], employee_data)
        os.makedirs(save_path, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        
        poses = [
            ("front", "Look directly at the camera", 2),
            ("left", "Turn your face to the left", 2),
            ("right", "Turn your face to the right", 2),
            ("up", "Tilt your face upwards", 2),
            ("down", "Tilt your face downwards", 2)
        ]
        
        all_images = []
        
        for pose, instruction, num_images in poses:
            images_captured = self.capture_pose(cap, pose, instruction, num_images, save_path, employee_data)
            all_images.extend(images_captured)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if all_images:
            employee_image = Image.open(all_images[0])  # Use the first image as profile photo
            self._register_employee(employee_institute_id, employee_name, employee_image, employee_data)
    
    def capture_pose(self, cap, pose, instruction, num_images, save_path, employee_data):
        print(f"\n{instruction}")
        print("Press 'c' when ready to capture.")
        
        images_captured = []
        while len(images_captured) < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
            
            faces = self.face_processor.detect_faces(frame)
            
            cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if len(faces) == 1:
                    image_path = os.path.join(save_path, f"{employee_data}_{pose}_{len(images_captured)+1}.jpg")
                    cv2.imwrite(image_path, frame)
                    images_captured.append(image_path)
                    print(f"Captured {pose} image {len(images_captured)}/{num_images}")
                else:
                    print("No face detected or multiple faces detected. Please try again.")
            elif key == ord('q'):
                break
        
        return images_captured
    
    def _register_employee(self, employee_institute_id, name, employee_image, employee_data):
        """Process captured images and save to database"""
        employee_folder = os.path.join(CONFIG["EMPLOYEE_DATA_ROOT"], employee_data)
        embeddings = []
        
        for image_file in os.listdir(employee_folder):
            image_path = os.path.join(employee_folder, image_file)
            image = cv2.imread(image_path)
            embeds = self.face_processor.get_embeddings(image)
            if embeds:
                embeddings.append(embeds[0])
        
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding)
            self.db.save_employee(employee_institute_id, name, avg_embedding, employee_image)
            print(f"Successfully registered {name}")
        else:
            print("Failed to generate embeddings. Please try registration again.")

# Usage
if __name__ == "__main__":
    registrar = EmployeeRegistrar()
    registrar.capture_face_samples()
