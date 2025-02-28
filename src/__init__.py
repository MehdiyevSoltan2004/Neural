from src.Face_Recognition import load_known_faces, recognize_faces_in_video

if __name__ == "__main__":
    known_faces_folder = "C:\\Users\\MSI\\Pictures\\Camera Roll\\"
    known_face_encodings, known_face_names = load_known_faces(known_faces_folder)
    recognize_faces_in_video(known_face_encodings, known_face_names)
