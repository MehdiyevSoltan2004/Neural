import cv2
import torch
import numpy as np
import os
from insightface.app import FaceAnalysis

KNOWN_FACES_DIR = "D:\\Python\\Neural\\data\\known_faces"
CONFIDENCE_THRESHOLD = 0.5 

device = "cuda" if torch.cuda.is_available() else "cpu"

face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0 if device == 'cuda' else -1)

def load_known_faces():
    known_faces = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(path)
        faces = face_analyzer.get(img)
        if faces:
            known_faces[filename] = faces[0].embedding  
            print(f"Loaded face: {filename}")
    return known_faces

known_faces = load_known_faces()
print(f"Total known faces: {len(known_faces)}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_analyzer.get(frame)
    
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        
        new_embedding = face.embedding
        
        label = "Unknown"
        color = (0, 0, 255)
        min_distance = float("inf")
        best_match = None
        
        for name, known_embedding in known_faces.items():
            similarity = np.dot(new_embedding, known_embedding) / (np.linalg.norm(new_embedding) * np.linalg.norm(known_embedding))

            dist = 1 - similarity
            
            if dist < min_distance:
                min_distance = dist
                best_match = name
        
        if best_match and min_distance < CONFIDENCE_THRESHOLD:
            label = best_match.split('.')[0]
            color = (0, 255, 0)
            print(f"Match found: {label}, distance: {min_distance:.3f}")
        else:
            print(f"Unknown face, best match distance: {min_distance:.3f}")
            
            if min_distance > 0.7:
                new_name = f"unknown_{len(known_faces) + 1}.jpg"
                face_img = frame[y1:y2, x1:x2].copy()
                cv2.imwrite(os.path.join(KNOWN_FACES_DIR, new_name), face_img)
                known_faces[new_name] = new_embedding
                print(f"Saved new face: {new_name}")
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({1-min_distance:.2f})", (x1, max(y1 - 10, 0)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()