from sklearn.metrics import classification_report, accuracy_score
import joblib

def evaluate(images, labels, model_path):
    print("Evaluating model")
    model = joblib.load(model_path)
    y_pred = model.predict(images)
    print(classification_report(labels, y_pred))
    print(f"Accuracy: {accuracy_score(labels, y_pred):.2f}")

def predict_plot(image, model):
    pass