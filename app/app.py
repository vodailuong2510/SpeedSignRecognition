import base64
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok
import joblib
import numpy as np
import cv2
from skimage.feature import hog

def hog_features(images):
    hog_features = []
    for i in range(len(images)):
        img = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        features, _ = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm="L2-Hys", visualize=True)
        hog_features.append(features)
    return np.array(hog_features)

def preprocessing(images, resize):
    images = [cv2.resize(img, resize) for img in images]
    images = [cv2.GaussianBlur(img, (5, 5), 0) for img in images]
    images = [cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) for img in images]
    return np.array(images)

model = joblib.load(r"C:\Users\vodai\Downloads\projects\SpeedSignRecognition\SVM_model.h5")

class_names = {
    0: "End of All Restrictions",
    1: "End of minimum speed limit 60 km/h",
    2: "End of minimum speed limit 80 km/h",
    3: "End of speed limit 40 km/h",
    4: "End of speed limit 50 km/h",
    5: "End of speed limit 60 km/h",
    6: "Minimum speed 60 km/h",
    7: "Minimum speed 80 km/h",
    8: "Speed limit 30 km/h",
    9: "Speed limit 40 km/h",
    10: "Speed limit 50 km/h",
    11: "Speed limit 60 km/h",
    12: "Speed limit 80 km/h"
}

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Sign Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            text-align: center;
            background: white;
            padding: 30px 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 400px;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        form {
            margin-top: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .file-input-wrapper {
            display: flex;
            justify-content: center;
            width: 100%;
            margin-bottom: 20px;
        }
        input[type="file"] {
            text-align: center;
            cursor: pointer;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        img {
            max-width: 100%;
            max-height: 300px;
            margin-top: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .resized {
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Traffic Sign Recognition</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <div class="file-input-wrapper">
                <input type="file" name="file" id="file" accept="image/*" required onchange="previewAndResize(event)">
            </div>
            <img id="resized" alt="" class="resized">
            <button type="submit">Upload and Predict</button>
        </form>
    </div>

    <script>
        function previewAndResize(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    // Resize the image
                    const canvas = document.createElement("canvas");
                    const ctx = canvas.getContext("2d");

                    const img = new Image();
                    img.onload = function() {
                        // Set desired resize dimensions
                        const resizeWidth = 128;
                        const resizeHeight = 128;

                        // Adjust canvas size
                        canvas.width = resizeWidth;
                        canvas.height = resizeHeight;

                        // Draw and resize the image
                        ctx.drawImage(img, 0, 0, resizeWidth, resizeHeight);

                        // Show the resized image
                        const resized = document.getElementById("resized");
                        resized.src = canvas.toDataURL();
                        resized.style.display = "block";
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    resized_image = cv2.resize(image, (128, 128))
    _, resized_buffer = cv2.imencode('.jpg', resized_image)
    resized_img_base64 = base64.b64encode(resized_buffer).decode('utf-8')

    image = preprocessing([image], resize=(128, 128))
    features = hog_features(image)

    predictions = model.predict(features)
    predicted_class = class_names[predictions[0]]

    result_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .container {{
                text-align: center;
                background: white;
                padding: 30px 40px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                color: #333;
                margin-bottom: 20px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin-bottom: 20px;
            }}
            p {{
                font-size: 18px;
                color: #555;
            }}
            a {{
                display: inline-block;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                margin-top: 20px;
                transition: background-color 0.3s ease;
            }}
            a:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Prediction Result</h1>
            <img src="data:image/jpeg;base64,{resized_img_base64}" alt="Resized Image" />
            <p><strong>Predicted Class:</strong> {predicted_class}</p>
            <a href="/">Go Back</a>
        </div>
    </body>
    </html>
    """

    return render_template_string(result_html)

if __name__ == "__main__":
    public_url = ngrok.connect(5000)  
    print(f"Public URL: {public_url}")
    app.run()
