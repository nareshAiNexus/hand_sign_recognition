
# Hand Gesture Recognition

This project uses OpenCV and Mediapipe to capture and recognize hand gestures in real-time using a webcam. The hand gestures are processed and classified using a KNN classifier trained on the extracted landmarks of the hand's position. The project allows you to save and label new gestures, which are stored in a CSV file for training purposes. The trained model can then be used to predict the class of a new hand gesture.

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.x
- OpenCV
- Mediapipe
- Scikit-learn

To install the required libraries, you can use the following command:

```bash
pip install opencv-python mediapipe scikit-learn
```

## Features

- **Real-time Hand Gesture Recognition:** The webcam captures hand gestures and displays the recognized class on the screen.
- **Gesture Labeling:** You can label and save new gestures using the keyboard (`s` to save, `q` to quit).
- **KNN Classifier:** The project uses a K-Nearest Neighbors (KNN) classifier for gesture recognition.
- **CSV Dataset:** The hand gesture data is stored in a CSV file, which can be used for training the KNN classifier.

## How to Run

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Install the Dependencies

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can manually install the dependencies by running:

```bash
pip install opencv-python mediapipe scikit-learn
```

### 3. Run the Python Script

Run the following Python script to start the hand gesture recognition program:

```bash
python app.py
```

### 4. Usage Instructions

- Press **`s`** to start saving the hand gesture points. After pressing `s`, you will be prompted to enter a class name (e.g., "fist", "peace", etc.).
- Press **`q`** to quit the program and stop capturing video.
- The dataset will be saved in the `hand_gesture_data` folder and can be used for training the KNN model.

### 5. Prediction

Once the dataset is saved, the KNN model will be used to predict the class of new gestures in real-time. The predicted class will be displayed on the screen.

## How the Model Works

1. **Data Collection:** The hand gestures are captured using a webcam and processed using the Mediapipe library to detect hand landmarks. Mediapipe detects 21 key points on the hand (fingertips, joints, etc.) and stores their 2D coordinates.
   
2. **Data Storage:** The hand landmarks (21 key points) for each gesture are saved in a CSV file, along with the class label that corresponds to the gesture.

3. **Model Training:** The KNN classifier is trained using the saved dataset. The 2D coordinates of the hand landmarks serve as input features, and the class labels represent the output classes.

4. **Gesture Recognition:** In real-time, the program captures a hand gesture, processes the landmarks, and uses the trained KNN model to predict the class of the gesture. The predicted class is displayed on the screen.

## File Structure

```
hand-gesture-recognition/
│
├── app.py   # Main Python script for gesture recognition
├── hand_gesture_data/            # Directory to store the gesture dataset
│   └── hand_gestures.csv         # CSV file containing the dataset (class label + hand landmarks)
├── requirements.txt             # List of dependencies
└── README.md                    # Project description and usage instructions
```

## Troubleshooting

- If the webcam is not displaying, make sure the webcam is properly connected and accessible.
- If the program fails to recognize gestures, ensure that your hand is visible in the webcam frame and that the lighting is good for gesture detection.

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments
