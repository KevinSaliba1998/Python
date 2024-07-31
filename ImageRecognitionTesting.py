import threading
import cv2
from deepface import DeepFace

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Global variables
counter = 0
face_match_results = []

# Load reference images
reference_images = {
    "UserName1": [
        cv2.imread(r"path_1")
    ],
    "UserName2": [
        cv2.imread(r"path_1")
    ]
}

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to check face against reference images
def check_face(face_crop, idx):
    global face_match_results
    face_match_results[idx] = "Unknown"
    for name, images in reference_images.items():
        for reference_img in images:
            try:
                if DeepFace.verify(face_crop, reference_img)['verified']:
                    face_match_results[idx] = name
                    return
            except ValueError:
                continue

# Function to draw box and name
def draw_box_and_name(frame, face_coords, name):
    (x, y, w, h) = face_coords
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if counter % 100 == 0:  # Adjust this value to balance accuracy and performance
        face_match_results = ["Unknown"] * len(faces)
        threads = []

        for i, (x, y, w, h) in enumerate(faces):
            face_crop = frame[y:y+h, x:x+w]
            # Check the DeepFace model's expected input size
            face_crop_resized = cv2.resize(face_crop, (224, 224))  # Resize to the expected input size for DeepFace
            thread = threading.Thread(target=check_face, args=(face_crop_resized, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    counter += 1

    for (x, y, w, h), name in zip(faces, face_match_results):
        draw_box_and_name(frame, (x, y, w, h), name)

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
