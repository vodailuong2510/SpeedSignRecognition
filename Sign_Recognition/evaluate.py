from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(images, labels, class_names, model_path):
    print("Evaluating model")
    model = joblib.load(model_path)
    y_pred = model.predict(images)
    print(classification_report(labels, y_pred))
    print(f"Accuracy: {accuracy_score(labels, y_pred):.2f}")

    cm = confusion_matrix(labels, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def predict_plot(image, model):
    pass