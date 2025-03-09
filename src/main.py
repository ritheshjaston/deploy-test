from flask import Flask, jsonify, request
from keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
import base64
import os

app = Flask(__name__)
CORS(app)

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'emotionaldetector.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_data = []

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    global emotion_data

    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data provided'}), 400

    # Decode the base64 frame data
    frame_data = base64.b64decode(data['frame'])
    np_frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            emotion_data.append(label)

            # Draw rectangle around the face and display label (optional)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Store emotions in a text file
    with open('detected_emotions.txt', 'w') as file:
        for emotion in emotion_data:
            file.write(f"{emotion}\n")

    return jsonify({'status': 'emotion detected', 'emotion': label})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(debug=True)
