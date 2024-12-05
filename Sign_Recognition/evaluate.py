import cv2
from sklearn.metrics import classification_report, accuracy_score
from .models import detect_sign
from .features import hog_features

def evaluate(images, labels, model):
    print("Evaluating model")
    y_pred = model.predict(images)
    print(classification_report(labels, y_pred))
    print(f"Accuracy: {accuracy_score(labels, y_pred):.2f}")

def predict(image, model, plot=False):
    contours, mask = detect_sign(image)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]

        try:
            features = hog_features(roi)
            label = model.predict([features])[0]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except:
            continue
    
    if plot:
        cv2.imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return image
