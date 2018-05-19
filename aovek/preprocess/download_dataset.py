import wget
import tarfile
import os


def download_dataset(url, path):
    try:
        os.mkdir(path)
    except OSError as e:
        pass

    filename = wget.download(url, path)

    tar = tarfile.open(filename, "r:gz")
    tar.extractall(path)
    tar.close()

    os.remove(filename)
