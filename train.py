import pickle
import json

from network import YOLO


def load_data(pickle_name):
    train_pickle_name = pickle_name['train']
    validation_pickle_name = pickle_name['validation']
    test_pickle_name = pickle_name['test']

    train_data, train_labels = get_data_from_pickle(train_pickle_name)
    validation_data, validation_labels =\
        get_data_from_pickle(validation_pickle_name)
    test_data, test_labels = get_data_from_pickle(test_pickle_name)

    return train_data, train_labels, validation_data, validation_labels,\
        test_data, test_labels


def get_data_from_pickle(pickle_name):
    with open(pickle_name, 'rb') as dsp:
        dataset = pickle.load(dsp)
        data = dataset['data']
        labels = dataset['labels']
        del dataset
    return data, labels


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    pickles_name = config['dataset']['pickle_name']

    network = YOLO(config)
    train_data, train_labels, validation_data, validation_labels,\
        test_data, test_labels = load_data(pickles_name)

    network.train(train_data, train_labels, validation_data, validation_labels,
                  test_data, test_labels)
