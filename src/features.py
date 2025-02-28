import face_recognition
import cv2
import os
import numpy as np

def load_known_faces(known_faces_folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_folder):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(known_faces_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
    
    return known_face_encodings, known_face_names

def recognize_faces_in_video(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])


        
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        
        print(f"Найдено лиц: {len(face_locations)}, координаты: {face_locations}")

        
        if not face_locations:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        
        print(f"Количество закодированных лиц: {len(face_encodings)}")

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


