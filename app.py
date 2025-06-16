from flask import Flask, request
import cv2
import numpy as np
import mediapipe as mp
import requests
import os

app = Flask(__name__)

FIREBASE_URL = "https://release-1-10b3a-default-rtdb.firebaseio.com/eye_status.json"
FIREBASE_AUTH = "2FUJYAuI6BFnpd9oDNSf67WKhi0s56dtFtpJNMNW"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 15

frame_counter = 0
last_state = -1

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def send_eye_status_to_firebase(state):
    try:
        data = {"eye": state}
        response = requests.put(f"{FIREBASE_URL}?auth={FIREBASE_AUTH}", json=data)
        print("✅ Sent to Firebase:", state)
    except Exception as e:
        print("🔥 Error sending to Firebase:", e)

@app.route("/upload", methods=["POST"])
def upload_image():
    global frame_counter, last_state

    if "image" not in request.files and not request.data:
        return "No image received", 400

    # Read image bytes (supporting raw binary or multipart)
    image_bytes = request.data if request.data else request.files["image"].read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return "Invalid image", 400

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            left_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
            right_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]

            left_ear = calculate_ear(np.array(left_eye))
            right_ear = calculate_ear(np.array(right_eye))
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= FRAME_THRESHOLD:
                    if last_state != 0:
                        send_eye_status_to_firebase(0)
                        last_state = 0
                    return "Sleeping", 200
            else:
                frame_counter = 0
                if last_state != 1:
                    send_eye_status_to_firebase(1)
                    last_state = 1
                return "Awake", 200
    else:
        frame_counter = 0
        return "No face detected", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
