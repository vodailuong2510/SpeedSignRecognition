import yaml
from SignBoard_Recognition.preprocessing import download, unzip, read_data
from SignBoard_Recognition.utils import plot_images

config_path = "./config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    print("Read configuration file successfully")

# download(config["data"]["link"])
# unzip(config["data"]["zip_path"], config["data"]["data_path"])

train_path = config["data"]["train_path"]
test_path = config["data"]["test_path"]
class_names = config["data"]["class_names"]

trainX, trainY = read_data(train_path)
testX, testY = read_data(test_path)

if config["output"]["plot"]:
    plot_images(trainX, trainY, class_names, title = "Train Images", num_images=5)
    # plot_images(testX, testY, class_names, title="Test Images", num_images=5)

