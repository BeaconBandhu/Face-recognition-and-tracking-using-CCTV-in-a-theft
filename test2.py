import cv2
import numpy as np
import mediapipe as mp
import serial

# Initialize serial communication with Arduino on COM6
ser1 = serial.Serial('COM6', 9600)  # Replace 'COM6' with the appropriate port for the motor

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

def calculate_angle(x_center, frameWidth):
    center_x = frameWidth / 2
    dx = x_center - center_x
    angle = (dx / frameWidth) * 180  # Adjust this calculation based on your needs
    return angle

# Load face detection model
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load age and gender models
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-25)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Find Iriun webcam
iriun_index = None
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 0:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Testing Iriun Webcam', frame)
            print("Is this the Iriun Webcam? Press y for Yes, n for No")
            key = cv2.waitKey(0)
            if key == ord('y'):
                iriun_index = i
                break
            cv2.destroyWindow('Testing Iriun Webcam')
    cap.release()

if iriun_index is None:
    print("Iriun Webcam not found.")
    exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture from Iriun webcam
cap = cv2.VideoCapture(iriun_index)
padding = 20
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from video. Exiting...")
        break

    frameHeight, frameWidth, _ = frame.shape
    frame, bboxs = faceBox(faceNet, frame)

    for bbox in bboxs:
        x_center = (bbox[0] + bbox[2]) // 2
        angle = calculate_angle(x_center, frameWidth)
        
        # Extract face region
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1), 
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender prediction
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        
        # Age prediction
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label = "{},{}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1) 
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the x-coordinate and angle on the image
        cv2.putText(frame, f'X: {x_center}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Angle: {angle:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Send the angle to Arduino
        ser1.write(f"{int(angle)}\n".encode())  # Send angle to the motor

    # Pose detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Visualize pose landmarks
        for landmark in landmarks:
            x = int(landmark.x * frameWidth)
            y = int(landmark.y * frameHeight)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        # Display angle between shoulder, elbow, and wrist
        angle = calculate_angle((shoulder[0] + elbow[0]) / 2, frameWidth)
        cv2.putText(frame, str(angle), (int(elbow[0] * frameWidth), int(elbow[1] * frameHeight)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Face and Pose Tracking', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser1.close()
