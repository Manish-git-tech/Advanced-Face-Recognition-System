import cv2
import os
from PIL import Image
import numpy as np
from database_handler import DatabaseManager
from config import CONFIG
from face_processor import FaceProcessor
import time
from datetime import datetime

class VisitorRegistrar:
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.db = DatabaseManager()

    def register_visitor(self, visitor_name=None, purpose_of_visit=None, allowed_time_str=None):
        """
        Interactive face registration for a visitor using the webcam.
        Prompts the user (or uses passed arguments) for:
         - visitor name,
         - purpose of visit,
         - allowed exit time (format: "YYYY-MM-DD HH:MM:SS").
        Captures a few face samples, averages the face embeddings, registers the visitor in
        the visitors table, and then logs the visitor's entry.
        """
        if not visitor_name:
            visitor_name = input("Enter visitor name: ").strip()
        if not purpose_of_visit:
            purpose_of_visit = input("Enter purpose of visit: ").strip()
        if not allowed_time_str:
            allowed_time_str = input("Enter allowed exit time (YYYY-MM-DD HH:MM:SS): ").strip()
        try:
            allowed_time = datetime.strptime(allowed_time_str, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print("Error parsing allowed exit time. Ensure the format is correct.")
            return

        # Create a unique folder name and path for the visitor samples.
        visitor_data = f"{visitor_name}_{int(time.time())}"
        save_path = os.path.join(CONFIG["VISITOR_DATA_ROOT"], visitor_data)
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(0)

        # For visitors, we'll capture a small set of images (front, left, right) for a quick registration.
        poses = [
            ("front", "Please look directly at the camera", 1),
            ("left", "Please turn your face to the left", 1),
            ("right", "Please turn your face to the right", 1)
        ]

        all_images = []

        for pose, instruction, num_images in poses:
            images_captured = self.capture_pose(cap, pose, instruction, num_images, save_path, visitor_data)
            all_images.extend(images_captured)

        cap.release()
        cv2.destroyAllWindows()

        if all_images:
            # Use the first captured image as the profile photo.
            visitor_image = Image.open(all_images[0])
            self._register_visitor(visitor_name, visitor_image, visitor_data, purpose_of_visit, allowed_time)
        else:
            print("No images captured. Registration failed.")

    def capture_pose(self, cap, pose, instruction, num_images, save_path, visitor_data):
        """
        Captures num_images for a given pose with an instruction on screen.
        Returns a list of file paths for the captured images.
        """
        print(f"\n{instruction}")
        print("Press 'c' to capture the image.")

        images_captured = []
        while len(images_captured) < num_images:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = self.face_processor.detect_faces(frame)

            cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow('Visitor Registration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if len(faces) == 1:
                    image_path = os.path.join(save_path, f"{visitor_data}_{pose}_{len(images_captured)+1}.jpg")
                    cv2.imwrite(image_path, frame)
                    images_captured.append(image_path)
                    print(f"Captured {pose} image {len(images_captured)}/{num_images}")
                else:
                    print("No face detected or multiple faces present. Please try again.")
            elif key == ord('q'):
                break

        return images_captured

    def _register_visitor(self, visitor_name, visitor_image, visitor_data, purpose_of_visit, allowed_time):
        """
        Processes captured visitor images, computes face embeddings,
        averages them, saves the visitor record, and logs the entry.
        """
        visitor_folder = os.path.join(CONFIG["VISITOR_DATA_ROOT"], visitor_data)
        embeddings = []

        for image_file in os.listdir(visitor_folder):
            image_path = os.path.join(visitor_folder, image_file)
            image = cv2.imread(image_path)
            embeds = self.face_processor.get_embeddings(image)
            if embeds:
                embeddings.append(embeds[0])

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding)
            # Save the visitor record; the embedding is stored as a BLOB.
            visitor_id = self.db.save_visitor(visitor_name, avg_embedding,
                                              purpose_of_visit, allowed_time,
                                              visitor_status="active")
            print(f"Successfully registered visitor: {visitor_name}")
            # Log the visitor's entry.
            self.db.log_visitor_entry(visitor_id)
        else:
            print("Failed to generate embeddings. Please try registration again.")

# Usage
if __name__ == "__main__":
    registrar = VisitorRegistrar()
    registrar.register_visitor()
