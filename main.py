import yaml
from Sign_Recognition.preprocessing import read_data, preprocessing_augment, over_sampling
from Sign_Recognition.utils import plot_images, download, unzip
from Sign_Recognition.models import SVC_training_with_GridSearch, RandomForest_training_with_GridSearch
from sklearn.model_selection import train_test_split
from Sign_Recognition.evaluate import evaluate
from Sign_Recognition.features import hog_features


config_path = "./config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    print("Read configuration file successfully")

# download(config["data"]["link"])
# unzip(config["data"]["zip_path"], config["data"]["extract_path"])

data_path = config["data"]["data_path"]
class_names = config["data"]["class_names"]
resize = config["data"]["resize"]
test_size = config["data"]["split_ratio"]["test_size"]

images, labels = read_data(data_path)

images, labels = preprocessing_augment(images, labels, resize, augment = True, weight = 3)

images, labels = over_sampling(images, labels, resize)

trainX, testX, trainY, testY = train_test_split(images, labels, test_size=test_size, random_state=22520834)

# if config["output"]["plot"]:
#     plot_images(trainX, trainY, class_names, title = "Train Images", num_images=10)
#     plot_images(testX, testY, class_names, title="Test Images", num_images=10)

train_features = hog_features(trainX)
test_features = hog_features(testX)

svc = SVC_training_with_GridSearch(train_features, trainY, config["output"]["SVM_save_path"])
evaluate(test_features, testY, config["output"]["SVM_save_path"])

rf = RandomForest_training_with_GridSearch(train_features, trainY, config["output"]["RF_save_path"])
evaluate(test_features, testY, config["output"]["RF_save_path"])