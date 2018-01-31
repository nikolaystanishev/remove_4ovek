import json
import wget
import tarfile
import os


def download_dataset():
    with open('./config.json') as config_file:
        config = json.load(config_file)

    url = config["dataset"]["url"]
    path = config["dataset"]["path"]

    try:
        os.mkdir(path)
    except OSError as e:
        pass

    filename = wget.download(url, path)

    tar = tarfile.open(filename, "r:gz")
    tar.extractall(path)
    tar.close()

    os.remove(filename)


if __name__ == '__main__':
    download_dataset()
