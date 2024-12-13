from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def SVC_training_with_GridSearch(images, labels, model_path):
    print("Start training with SVM Random Forest using Grid Search")

    param_grid = {
        'C': [0.1, 1, 10],             
        'gamma': ['scale', 'auto'],     
        'kernel': ['poly', 'rbf']             
    }

    clf = SVC()

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(images, labels)

    print(f"Best parameters: {grid_search.best_params_}")
    print("Finish training")

    joblib.dump(grid_search.best_estimator_, model_path)
    return grid_search.best_estimator_

def RandomForest_training_with_GridSearch(images, labels, model_path):
    print("Start training with Random Forest Random Forest using Grid Search")
    
    param_grid = {
        'n_estimators': [50, 100, 200],        
        'max_depth': [10, 20, 30],       
        'min_samples_split': [2, 5, 10],     
        'min_samples_leaf': [1, 2, 4],        
        'class_weight': ['balanced', 'balanced_subsample']
    }

    rf = RandomForestClassifier(random_state=22520834, bootstrap=True)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(images, labels)

    print(f"Best parameters: {grid_search.best_params_}")
    print("Finish training")

    joblib.dump(grid_search.best_estimator_, model_path)
    return grid_search.best_estimator_
