import cv2
from sklearn.metrics import classification_report, accuracy_score

def evaluate(images, labels, model):
    print("Evaluating model")
    y_pred = model.predict(images)
    print(classification_report(labels, y_pred))
    print(f"Accuracy: {accuracy_score(labels, y_pred):.2f}")

def predict_plot(image, model):
    pass