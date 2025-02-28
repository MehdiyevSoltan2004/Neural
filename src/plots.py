import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from insightface.app import FaceAnalysis

KNOWN_FACES_DIR = "D:\\Python\\Neural\\data\\known_faces"
CONFIDENCE_THRESHOLD = 0.4
DETECTION_THRESHOLD = 0.6
MIN_FACE_SIZE = 120  # Минимальный размер лица в пикселях

device = "cuda" if torch.cuda.is_available() else "cpu"
detector = YOLO("D:\\Python\\Neural\\models\\yolov8n-face-lindevs.pt").to(device)

# Используем CPU для стабильности
face_recognizer = FaceAnalysis(name='buffalo_l', root='insightface_db')
face_recognizer.prepare(ctx_id=-1)  # -1 = CPU

def load_known_faces():
    known_faces = {}
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Поврежденное изображение: {filename}")
                continue
            
            # Проверяем размер изображения
            h, w, _ = img.shape
            if w < 100 or h < 100:
                print(f"Слишком маленькое изображение: {filename} ({w}x{h})")
                continue
            
            # Анализируем с автоматическим выравниванием
            faces = face_recognizer.get(img)
            if faces:
                known_faces[filename] = faces[0].embedding
        except Exception as e:
            print(f"Ошибка обработки {filename}: {e}")
    
    print("Загруженные лица:", list(known_faces.keys()))
    return known_faces

known_faces = load_known_faces()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector(frame, conf=DETECTION_THRESHOLD, verbose=False)
    
    for result in results:
        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            continue
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        for (box, conf) in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            
            # Проверяем размеры лица
            w, h = x2 - x1, y2 - y1
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE or conf < DETECTION_THRESHOLD:
                continue
            
            face_img = frame[y1:y2, x1:x2].copy()
            
            try:
                # Проверяем критерии качества изображения
                if face_img.size == 0:
                    print("Пустое изображение лица")
                    continue
                
                # Применяем гауссовый блюр для уменьшения шума
                face_img = cv2.GaussianBlur(face_img, (5, 5), 0)
                
                # Пытаемся найти лицо с выравниванием
                faces = face_recognizer.get(face_img)
                
                if not faces:
                    print("Не удалось выровнять лицо (1)")
                    continue
                
                face = faces[0]
                
                # Проверяем наличие ключевых точек
                if face.bbox is None or face.kps is None:
                    print("Не удалось выровнять лицо (2)")
                    continue
                
                # Получаем выровненное лицо
                aligned_face = face_recognizer.get(face_img, max_num=1)
                if not aligned_face:
                    print("Не удалось выровнять лицо (3)")
                    continue
                
                embedding = aligned_face[0].embedding
                
                label = "Unknown"
                color = (0, 0, 255)
                
                min_dist = float("inf")
                best_match = None
                
                for name, known_emb in known_faces.items():
                    dist = np.linalg.norm(embedding - known_emb)
                    print(f"Сравнение с {name}: {dist:.4f}")
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match = name
                
                if best_match and min_dist < CONFIDENCE_THRESHOLD:
                    label = os.path.splitext(best_match)[0]
                    color = (0, 255, 0)
                else:
                    new_name = f"unknown_{len(known_faces)+1}.jpg"
                    cv2.imwrite(os.path.join(KNOWN_FACES_DIR, new_name), face_img)
                    known_faces[new_name] = embedding
                    print(f"Добавлено новое лицо: {new_name}")
                
                # Отрисовка рамки и метки
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            except Exception as e:
                print(f"Ошибка обработки: {e}")
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()