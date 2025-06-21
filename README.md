# Speed Sign Recognition 🚦

A machine learning-based traffic speed sign recognition system that can classify various speed limit signs using computer vision and machine learning techniques.

## 🌟 Key Features

- **Traffic Sign Classification**: Recognizes 13 different types of speed limit signs
- **Machine Learning Models**: Support Vector Machine (SVM) and Random Forest classifiers
- **Feature Extraction**: Histogram of Oriented Gradients (HOG) feature extraction
- **Data Augmentation**: Image preprocessing and augmentation for better model performance
- **Web Interface**: User-friendly Flask web application for real-time predictions
- **Model Evaluation**: Comprehensive evaluation metrics and visualization tools

## 🏗️ System Architecture

```
SpeedSignRecognition/
├── main.py                 # Main training script
├── app/
│   └── app.py             # Flask web application
├── Sign_Recognition/      # Core ML module
│   ├── preprocessing.py   # Data preprocessing and augmentation
│   ├── features.py        # HOG feature extraction
│   ├── models.py          # ML model training (SVM, Random Forest)
│   ├── evaluate.py        # Model evaluation and metrics
│   └── utils.py           # Utility functions
├── data/                  # Dataset storage
│   ├── images/           # Traffic sign images
│   └── labels.txt        # Image labels
├── config.yaml           # Configuration file
└── requirements.txt      # Python dependencies
```

## 🚦 Supported Traffic Signs

The system can recognize the following 13 speed limit signs:

1. **End of All Restrictions**
2. **End of Minimum Speed Limit 60km/h**
3. **End of Minimum Speed Limit 80km/h**
4. **End of Maximum Speed Limit 40km/h**
5. **End of Maximum Speed Limit 50km/h**
6. **End of Maximum Speed Limit 60km/h**
7. **Minimum Speed Limit 60km/h**
8. **Minimum Speed Limit 80km/h**
9. **Maximum Speed Limit 30km/h**
10. **Maximum Speed Limit 40km/h**
11. **Maximum Speed Limit 50km/h**
12. **Maximum Speed Limit 60km/h**
13. **Maximum Speed Limit 80km/h**

## 🛠️ Technologies Used

### Machine Learning & Computer Vision
- **OpenCV**: Image processing and computer vision
- **Scikit-learn**: Machine learning algorithms (SVM, Random Forest)
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization

### Web Framework
- **Flask**: Web application framework
- **Pyngrok**: Secure tunneling for web access

### Data Processing
- **Albumentations**: Image augmentation library
- **Imbalanced-learn**: Handling imbalanced datasets
- **Scikit-image**: Image processing utilities

### Development Tools
- **Jupyter**: Interactive development
- **TensorFlow/Keras**: Deep learning capabilities (optional)
- **PyYAML**: Configuration management

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd SpeedSignRecognition
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
# The dataset will be automatically downloaded when running main.py
# or you can manually download from the Google Drive link in config.yaml
```

## ⚙️ Configuration

The `config.yaml` file contains all configuration parameters:

```yaml
data:
  link: "https://docs.google.com/uc?export=download&id=1YGQLQyMgwjTTIcIPv7Cj31O2BX1Fhl5X"
  data_path: "./data"
  split_ratio:
    test_size: 0.3
  class_names:
    0: "End of All Restrictions"
    1: "End of Minimum Speed Limit 60km/h"
    # ... more classes
  resize: [128, 128]

output:
  SVM_save_path: "./SVM_model.h5"
  RF_save_path: "./RF_model.pkl"
```

## 📚 Usage

### 1. Training the Models

Run the main training script:

```bash
python main.py
```

This will:
- Download and extract the dataset
- Preprocess and augment the images
- Extract HOG features
- Train SVM and Random Forest models
- Evaluate model performance
- Save trained models

### 2. Web Application

Start the Flask web application:

```bash
python app/app.py
```

The web interface will be available at:
- **Local**: `http://localhost:5000`
- **Public**: The app will automatically create a public URL using ngrok

### 3. Using the Web Interface

1. Open the web application in your browser
2. Upload an image of a traffic speed sign
3. The system will process the image and display the prediction
4. Results show the predicted sign type and confidence

## 🔧 Development

### Project Structure

- **`main.py`**: Main training pipeline
- **`app/app.py`**: Flask web application
- **`Sign_Recognition/`**: Core machine learning module
  - `preprocessing.py`: Data preprocessing and augmentation
  - `features.py`: HOG feature extraction
  - `models.py`: Model training functions
  - `evaluate.py`: Evaluation metrics
  - `utils.py`: Utility functions

### Key Functions

#### Data Preprocessing
```python
from Sign_Recognition.preprocessing import read_data, preprocessing_augment, over_sampling

# Load and preprocess data
images, labels = read_data(data_path)
images, labels = over_sampling(images, labels, resize)
images, labels = preprocessing_augment(images, labels, resize, augment=True)
```

#### Feature Extraction
```python
from Sign_Recognition.features import hog_features

# Extract HOG features
features = hog_features(images)
```

#### Model Training
```python
from Sign_Recognition.models import SVC_training_with_GridSearch, RandomForest_training_with_GridSearch

# Train SVM model
svc = SVC_training_with_GridSearch(train_features, trainY, save_path)

# Train Random Forest model
rf = RandomForest_training_with_GridSearch(train_features, trainY, save_path)
```

#### Model Evaluation
```python
from Sign_Recognition.evaluate import evaluate

# Evaluate model performance
evaluate(test_features, testY, class_names, model_path)
```

## 📊 Model Performance

The system uses two machine learning approaches:

### Support Vector Machine (SVM)
- **Advantages**: Good for high-dimensional data, effective with HOG features
- **Use Case**: Primary classification model
- **Performance**: Optimized with GridSearch for hyperparameter tuning

### Random Forest
- **Advantages**: Robust, handles non-linear relationships well
- **Use Case**: Alternative classification model
- **Performance**: Ensemble method with multiple decision trees

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for each class
- **Recall**: Recall for each class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🎯 Data Processing Pipeline

1. **Data Loading**: Load images and labels from dataset
2. **Over-sampling**: Balance dataset classes
3. **Preprocessing**: Resize, normalize, and apply Gaussian blur
4. **Augmentation**: Apply transformations to increase dataset size
5. **Feature Extraction**: Extract HOG features from images
6. **Model Training**: Train SVM and Random Forest models
7. **Evaluation**: Assess model performance on test set

## 🔍 Feature Extraction

The system uses **Histogram of Oriented Gradients (HOG)** for feature extraction:

- **Orientations**: 9 gradient orientations
- **Pixels per cell**: 8x8 pixels
- **Cells per block**: 2x2 cells
- **Block normalization**: L2-Hys normalization

HOG features capture the local gradient information, making them effective for traffic sign recognition.

## 🚀 Deployment

### Local Deployment
```bash
# Train models
python main.py

# Start web application
python app/app.py
```

### Production Deployment
For production deployment, consider:
- Using a production WSGI server (Gunicorn, uWSGI)
- Setting up proper logging
- Implementing error handling
- Adding authentication if needed
- Using HTTPS for security

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Project Link**: [https://github.com/your-username/SpeedSignRecognition](https://github.com/your-username/SpeedSignRecognition)
- **Email**: vodailuong2510@gmail.com

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Scikit-learn team for machine learning algorithms
- Flask framework for web development
- Traffic sign dataset contributors

## 📚 References

- Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection
- Support Vector Machines for classification
- Random Forest ensemble methods
- Traffic sign recognition best practices
