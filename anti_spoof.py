import cv2
import dlib
from scipy.spatial import distance
from skimage.feature import local_binary_pattern
import numpy as np

class LivenessDetection:
    def _init_(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        self.prev_nose_x = None
        self.blink_counter = 0
        self.blink_state = False  # Track eye state for better accuracy
        self.blink_threshold = 2   # Number of blinks required for real face
        
        self.texture_scores = []
        self.real_frames = 0
        self.fake_frames = 0

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def detect_blink(self, landmarks):
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < 0.2 and not self.blink_state:  # Eyes closed
            self.blink_counter += 1
            self.blink_state = True
        elif ear > 0.25:  # Eyes open again
            self.blink_state = False

    def detect_movement(self, landmarks):
        nose_x = landmarks.part(30).x  # Nose tip
        
        if self.prev_nose_x is not None and abs(nose_x - self.prev_nose_x) > 10:
            self.prev_nose_x = nose_x
            return True  # Movement detected

        self.prev_nose_x = nose_x
        return False

    def analyze_texture(self, gray):
        lbp = local_binary_pattern(gray, P=24, R=8, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), density=True)
        texture_score = np.sum(hist[:10])

        self.texture_scores.append(texture_score)
        if len(self.texture_scores) > 5:
            self.texture_scores.pop(0)  # Maintain rolling average

        avg_texture_score = np.mean(self.texture_scores)
        return avg_texture_score < 0.5  # Low variation means possible spoof

    def is_real_face(self, gray, landmarks):
        self.detect_blink(landmarks)
        movement_detected = self.detect_movement(landmarks)
        texture_real = self.analyze_texture(gray)

        # Real face must blink at least twice and show movement & natural texture
        return (self.blink_counter >= self.blink_threshold) and movement_detected and texture_real


# *Start Webcam Live Feed*
cap = cv2.VideoCapture(0)
detector = LivenessDetection()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detector(gray)

    if len(faces) == 0:
        label = "❌ No Face Detected"
        color = (0, 0, 255)
    else:
        for face in faces:
            landmarks = detector.predictor(gray, face)
            is_real = detector.is_real_face(gray, landmarks)

            # Stabilization: Consistent results over 10 frames
            if is_real:
                detector.real_frames += 1
                detector.fake_frames = 0
            else:
                detector.fake_frames += 1
                detector.real_frames = 0

            if detector.real_frames >= 10:
                label = "✅ Real Face"
                color = (0, 255, 0)
            elif detector.fake_frames >= 10:
                label = "🚨 Fake Face"
                color = (0, 0, 255)
            else:
                label = "⏳ Analyzing..."
                color = (255, 255, 0)

    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()