import os
import urllib.request

DATA_DIR = "data"
BASE_URL = "https://s3-eu-west-1.amazonaws.com/kate-datasets/london_smartmeters/"
TRAIN_FILENAME = "train.zip"
TEST_FILENAME = "test.zip"

if __name__ == "__main__":

    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    for file_ in [TRAIN_FILENAME, TEST_FILENAME]:
        url = BASE_URL + file_
        req = urllib.request.urlopen(url)
        data = req.read()

        with open(os.path.join(DATA_DIR, file_), "wb") as f:
            f.write(data)
