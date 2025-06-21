from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)
model = load_model("D:\DL_mini_project\sign_language_model(2).keras")  # Make sure this is the path to your trained model

# Define the class labels
class_labels = [chr(i) for i in range(65, 91)] + ['space','nothing'] # A to Z

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image part in the request', 400

    file = request.files['image']

    if file.filename == '':
        return 'No selected file', 400

    # Read and preprocess image
    #image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    image = cv2.resize(image, (64, 64))
    image = image.reshape(1, 64, 64, 1)  # (batch_size, height, width, channels)
    image = image.astype('float32') / 255.0
    # Predict
    prediction = model.predict(image)
    print("Predicted probabilities:", prediction)
    predicted_class = class_labels[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
