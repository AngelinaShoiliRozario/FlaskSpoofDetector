import cv2
import numpy as np
import face_recognition

# Function to compute face embeddings
def compute_face_embeddings(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return face_encodings

# Function to check if faces match using embeddings
def check_face_match(embeddings1, embeddings2):
    if len(embeddings1) == 0 or len(embeddings2) == 0:
        return False
    
    for encoding1 in embeddings1:
        for encoding2 in embeddings2:
            # Compare embeddings using L2 distance
            distance = np.linalg.norm(encoding1 - encoding2)
            # You may need to adjust the threshold depending on your use case
            if distance < 0.6:  # Threshold for considering faces as a match
                return True
    return False

cap = cv2.VideoCapture(0)
prev_embeddings = None  

while True:
    ret, frame = cap.read()

    if ret:
        current_embeddings = compute_face_embeddings(frame)

        if prev_embeddings is None:
            prev_embeddings = current_embeddings
        else:
            if check_face_match(current_embeddings, prev_embeddings):
                cv2.putText(frame, 'MATCH!', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'NO MATCH!', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("VIDEO", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            prev_embeddings = current_embeddings

# Release the VideoCapture object and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
