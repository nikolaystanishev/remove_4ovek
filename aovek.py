import argparse
import json
from PIL import Image

from aovek.preprocess.download_dataset import download_dataset
from aovek.preprocess.cvpr10_processing import CVPR10Processing
from aovek.preprocess.voc_processing import VOCProcessing
from aovek.training.train import Train
from aovek.visualization.predict import Predict
from aovek.validate.eval_metrics import EvalMetrics
from aovek.video.video_to_image import VideoToImage


parser = argparse.ArgumentParser(description='''
4ovek:
    File for contolling people detection process
''')
parser._action_groups.pop()

required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-config_file', help='Path to config file.',
                      required=True)

optional.add_argument('-dataset_download', help='Download dataset.',
                      action='store_true')
optional.add_argument('-processes_dataset', help='Processes dataset.',
                      action='store_true')
optional.add_argument('-train', help='Train convolutional neural network.',
                      action='store_true')
optional.add_argument('-predict', help='Make predictions for entire dataset.',
                      action='store_true')
optional.add_argument('-evaluate', help='Evaluate trained model.',
                      action='store_true')
optional.add_argument('-process_video', metavar='VIDEO',
                      help='Make photo without people from video')


def dataset_download(config):
    download_dataset(config)


def processes_dataset(config):
    if 'cvpr10' in config['dataset']['dataset']:
        dp = CVPR10Processing(config)
    elif 'voc' in config['dataset']['dataset']:
        dp = VOCProcessing(config)

    dp.pickle_dataset()

    print('Taken time: {}'.format(str(dp.get_time())))


def train(config):
    train = Train(config)
    train.load_dataset()
    train.train(config)


def predict(config):
    predict = Predict(config)
    predict.make_predictions_for_datasets()


def evaluate(config):
    eval_metrics = EvalMetrics(config)
    eval_metrics.eval_pickles_metrics()


def process_video(config, video_path):
    video_processing = VideoToImage(config)
    image_array = video_processing.process_video_file(video_path)

    image_filename = video_path.split('/')[-1].rsplit('.', 1)[0] + '.png'

    image = Image.fromarray(image_array)
    image.save(image_filename)


if __name__ == '__main__':
    args = parser.parse_args()

    config_file = args.config_file

    with open(config_file) as c_f:
        config = json.load(c_f)

    if args.dataset_download:
        dataset_download(config)
    elif args.processes_dataset:
        processes_dataset(config)
    elif args.train:
        train(config)
    elif args.predict:
        predict(config)
    elif args.evaluate:
        evaluate(config)
    elif args.process_video:
        process_video(config, args.process_video)
