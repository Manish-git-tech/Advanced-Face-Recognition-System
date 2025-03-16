# face_processor.py
from config import CONFIG
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceProcessor:
    def __init__(self):
        self.app = FaceAnalysis(
            name=CONFIG["FACE_MODEL_NAME"],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.embedding_history_size = 10
        self.original_weight = 0.95  # Weight given to the original embedding
    
    def get_embeddings(self, image):
        """Extract face embeddings from an image"""
        faces = self.app.get(image)
        return [face.embedding for face in faces] if faces else []
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(frame, f"Conf: {face.det_score:.2f}", 
                    (bbox[0]+10, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
        return [face for face in faces 
            if face.det_score > CONFIG["FACE_DETECTION_CONFIDENCE"]]

    def update_embedding(self, employee):
        original_embedding = employee['original_encoding']
        new_embedding = employee['current_embedding']
        
        if 'embedding_history' not in employee:
            employee['embedding_history'] = [original_embedding]
        
        employee['embedding_history'].append(new_embedding)
        if len(employee['embedding_history']) > self.embedding_history_size:
            employee['embedding_history'] = employee['embedding_history'][-self.embedding_history_size:]
        
        # Compute weighted average
        recent_embeddings = np.array(employee['embedding_history'])
        weighted_avg = (self.original_weight * original_embedding + 
                        (1 - self.original_weight) * np.mean(recent_embeddings, axis=0))
        
        return weighted_avg / np.linalg.norm(weighted_avg)  # Normalize the updated embedding

    def reset_embedding(self, employee):
        employee['encoding'] = employee['original_encoding']
        employee['embedding_history'] = [employee['original_encoding']]