"""
Predict a whole dataset
"""

import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model


def main():
    """These are the main predicting settings. Set each before running
    this file."""

    if (len(sys.argv) == 7):
        seq_length = int(sys.argv[1])
        class_limit = int(sys.argv[2])
        saved_model = sys.argv[3]
        video_path = sys.argv[4]
        image_height = int(sys.argv[5])
        image_width = int(sys.argv[6])
    else:
        print ("Usage: python predict.py sequence_length class_limit saved_model_name video_directory")
        # TODO: how do you specify the dir ?
        print ("Example: python predict.py 75 2 lstm-features.095-0.090.hdf5 /example_dir 720 1280")
        exit (1)

    sequences_dir = os.path.join('/content/drive/MyDrive/cnn/predict/data', 'sequences')
    if not os.path.exists(sequences_dir):
        os.mkdir(sequences_dir)

    checkpoints_dir = os.path.join('/content/drive/MyDrive/cnn/predict/data', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # model can be only 'lstm'
    model = 'lstm'
    saved_model = None  # None or weights file
    load_to_memory = False # pre-load the sequences into memory
    batch_size = 1
    nb_epoch = 1
    data_type = 'features'
    image_shape = (image_height, image_width, 3)

    extract_features(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)
    predict(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
