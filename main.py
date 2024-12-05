import yaml
from Sign_Recognition.preprocessing import download, unzip, read_data, preprocessing_images
from Sign_Recognition.utils import plot_images
from Sign_Recognition.features import roi_features
from Sign_Recognition.models import SVC_training

config_path = "./config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    print("Read configuration file successfully")

# download(config["data"]["link"])
# unzip(config["data"]["zip_path"], config["data"]["data_path"])

train_path = config["data"]["train_path"]
test_path = config["data"]["test_path"]
class_names = config["data"]["class_names"]

trainX, trainY = read_data(train_path, config["data"]["resize"])
# testX, testY = read_data(test_path, config["data"]["resize"])

trainX, testX = preprocessing_images(trainX), preprocessing_images(testX)

if config["output"]["plot"]:
    plot_images(trainX, trainY, class_names, title = "Train Images", num_images=5)
    # plot_images(testX, testY, class_names, title="Test Images", num_images=5)

