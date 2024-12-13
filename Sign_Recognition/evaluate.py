from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(images, labels, class_names, model_path):
    print("Evaluating model")
    model = joblib.load(model_path)
    y_pred = model.predict(images)
    print(classification_report(labels, y_pred))

    cm = confusion_matrix(labels, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def predict_plot(image, model):
    pass