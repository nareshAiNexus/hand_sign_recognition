import cv2
import mediapipe as mp
import csv
import os
from sklearn.neighbors import KNeighborsClassifier

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Directory to store the dataset
dataset_dir = "hand_gesture_data"
os.makedirs(dataset_dir, exist_ok=True)
csv_file_path = os.path.join(dataset_dir, "hand_gestures.csv")

# Create CSV file for storing points
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])

# Load dataset for prediction
X, y = [], []
if os.path.exists(csv_file_path):
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            y.append(row[0])
            X.append([float(value) for value in row[1:]])

# Train a KNN classifier if data exists
knn = None
if X and y:
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

# Start capturing video
cap = cv2.VideoCapture(0)
print("Press 's' to start saving points, 'q' to quit.")

saving_points = False
current_class = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the image for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract hand points
            landmarks = hand_landmarks.landmark
            points = [(landmark.x, landmark.y) for landmark in landmarks]

            # Display hand points
            for i, (x, y) in enumerate(points):
                height, width, _ = frame.shape
                cx, cy = int(x * width), int(y * height)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Predict the class name if a model exists
            if knn:
                flattened_points = [coord for point in points for coord in point]
                predicted_class = knn.predict([flattened_points])[0]
                cv2.putText(frame, f"Class: {predicted_class}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    # Handle key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if not saving_points:
            current_class = input("Enter class for this gesture: ")
            saving_points = True
            print(f"Started saving points for class '{current_class}'.")
        else:
            # Save the points when 's' is pressed
            flattened_points = [current_class] + [coord for point in points for coord in point]
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(flattened_points)
            print(f"Saved points with class '{current_class}'.")
            saving_points = False
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
