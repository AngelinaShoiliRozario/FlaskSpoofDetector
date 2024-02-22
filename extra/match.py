import cv2
import numpy as np
import mediapipe as mp
import face_recognition

def check_face(frame):
    image1_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    # Detect faces in both images
    face_locations1 = face_recognition.face_locations(image1_rgb)
    face_locations2 = face_recognition.face_locations(image2_rgb)
    # Encode faces
    face_encodings1 = face_recognition.face_encodings(image1_rgb, face_locations1)
    face_encodings2 = face_recognition.face_encodings(image2_rgb, face_locations2)

    # Compare faces
    for encoding1 in face_encodings1:
        for encoding2 in face_encodings2:
            results = face_recognition.compare_faces([encoding1], encoding2)
            print(results[0])
            if results[0] == True:
                return True
            else:
                return False

cap = cv2.VideoCapture(0)
cap
prev_frame = None  
required_pose = 'right'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

while True:

    ret, image = cap.read()
    

    np_image = np.array(image)
        
    if np_image.shape[2] == 3:  # RGB image
        image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    
    image= cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)  # convert to grayscale
    # image.flags.writeable = False

    results = face_mesh.process(image)

    img_h, img_w, img_c = image.shape

    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # print(idx)
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x + (img_w/2), lm.y * img_h)
                        node_3d = (lm.x + img_w, lm.y * img_h, lm.z * 3000)

                    x,y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x,y])

                    face_3d.append([x,y,lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                [0, focal_length, img_w/2],
                [0,0,1]
            ])
            dist_matrix = np.zeros((4,1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            # print(angles)

            if y < -10:
                text = 'left'
            elif y > 10 and y<16:
                text = 'right'
            elif x < -10:
                text = 'down'
            elif x > 10:
                text = 'up'
            elif x < 5 and x > (-5) and y < 5 and y > (-5):
                text ='forward'
                if prev_frame is None: 
                    # cv2.putText(image, f'Need to capture photo', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    print('Frame made ')
                    prev_frame = image.copy()
            else:
                text = 'forward'
            matchNum = True
            if(required_pose == text):
                print(f'in function detected pose {text}')
            if prev_frame is not None: 
                # if x < 15 and x > (-15) and y < 15 and y > (-10):
                checkFace = check_face(image)
                if checkFace:
                    cv2.putText(image, f'MATCH! X:{round(x)} Y:{round(y)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    matchNum = False
                    cv2.putText(image, f'NO MATCH!  X:{round(x)} Y:{round(y)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    
    cv2.imshow('Frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to exit
        break


cap.release()
cv2.destroyAllWindows()