from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def SVC_training_with_GridSearch(images, labels, model_path):
    print("Start training with SVM")

    param_grid = {
        'C': [0.1, 1, 10],             
        'gamma': ['scale', 'auto'],     
        'kernel': ['linear', 'poly', 'rbf']             
    }

    clf = SVC()

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(images, labels)

    print(f"Best parameters: {grid_search.best_params_}")
    print("Finish training")

    joblib.dump(rid_search.best_estimator_, model_path)
    return grid_search.best_estimator_

def RandomForest_training_with_GridSearch(images, labels, model_path):
    print("Start training with Random Forest using GridSearch")
    
    param_grid = {
        'n_estimators': [50, 100, 200],        
        'max_depth': [None, 10, 20, 30],       
        'min_samples_split': [2, 5, 10],     
        'min_samples_leaf': [1, 2, 4],        
        'max_features': ['auto', 'sqrt', 'log2'],  
    }

    rf = RandomForestClassifier(random_state=22520834)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(images, labels)

    print(f"Best parameters: {grid_search.best_params_}")
    print("Finish training")

    joblib.dump(model_path, 'rf_model.pkl')
    return grid_search.best_estimator_