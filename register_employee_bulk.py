import json
import os
import cv2
import numpy as np
from PIL import Image
from config import CONFIG
from face_processor import FaceProcessor
from database_handler import DatabaseManager
import time

class BulkEmployeeRegistrar:
    def __init__(self):
        # Reuse the same face processor and database handler
        self.face_processor = FaceProcessor()
        self.db = DatabaseManager()
    
    def register_employees_from_file(self, file_path):
        """
        Reads a JSON file containing employee information and registers each employee.
        The JSON is expected to have an "employees" key with a list of employee records.
        Each record must include:
            - name: employee name (string)
            - employee_institute_id: unique employee id (string)
            - photos: list of file paths to the employee's photos
        """
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        employees = data.get("employees", [])
        for emp in employees:
            name = emp["name"]
            employee_institute_id = emp["employee_institute_id"]
            photos = emp["photos"]  # List of photo paths
            
            # Create a unique folder to store (or reference) these images if needed
            employee_data = f"{name}_{employee_institute_id}"
            save_path = os.path.join(CONFIG["EMPLOYEE_DATA_ROOT"], employee_data)
            os.makedirs(save_path, exist_ok=True)
            
            embeddings = []
            for photo_path in photos:
                # Read the image from the given path.
                image = cv2.imread(photo_path)
                if image is None:
                    print(f"Warning: Unable to read {photo_path} for employee {name}.")
                    continue
                # Optionally, copy the file into the save_path if you wish:
                # dest_path = os.path.join(save_path, os.path.basename(photo_path))
                # cv2.imwrite(dest_path, image)
                
                # Extract embedding using your face processor.
                emb_list = self.face_processor.get_embeddings(image)
                if emb_list:
                    embeddings.append(emb_list[0])
            
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding /= np.linalg.norm(avg_embedding)
                # Use the first photo as the profile photo
                try:
                    profile_photo = Image.open(photos[0])
                except Exception as e:
                    print(f"Error opening profile photo for {name}: {e}")
                    continue
                self.db.save_employee(employee_institute_id, name,
                                      avg_embedding, profile_photo)
                print(f"Successfully registered {name} ({employee_institute_id}).")
            else:
                print(f"Failed to generate embeddings for {name} ({employee_institute_id}).")
