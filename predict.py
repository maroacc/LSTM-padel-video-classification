"""
Predict a whole dataset
"""

import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from extract_features import extract_features
from keras.models import load_model
from models import ResearchModels
from results import Results


def predict(data_type, seq_length, model, video_path, saved_model=None,
            class_limit=None, image_shape=None,
            load_to_memory=False, batch_size=32, nb_epoch=100, train_test='train'):
    # Get the data and process it.
    global class_indices, class_indices
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.6) // batch_size

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    generator_predict_train = data.frame_generator_predict(1, train_test, data_type)
    classes, filenames = data.get_classes_predict(1, train_test, data_type)
    # print('generator_predict_train:')
    # print(next(generator_predict_train))
    prediction_train = rm.model.predict_generator(generator_predict_train)
    print('Predictions:')
    print(prediction_train)
    print('classes')
    print(classes)
    print('filenames')
    print(filenames)

    # Format results and compute classification statistics
    dataset_name = 'THETIS 2-classes'
    class_indices = {"backhand": 0, "forehand": 1}
    # class_indices = {"backhand": 0, "backhand2hands": 1, "backhand_slice": 2, "backhand_volley": 3, "flat_service": 4,
    #                  "forehand_flat": 5, "forehand_openstands": 6, "forehand_slice": 7, "forehand_volley": 8,
    #                  "kick_service": 9, "slice_service": 10, "smash": 11}
    predicted_labels = np.argmax(prediction_train, axis=1).ravel().tolist()
    print('Predicted labels:')
    print(predicted_labels)
    results = Results(class_indices, dataset_name=dataset_name)
    accuracy, confusion_matrix, classification = results.compute(filenames, classes, predicted_labels)
    # Display and save results
    results.print(accuracy, confusion_matrix)
    results.save(confusion_matrix, classification, prediction_train, saved_model, train_test)


def main():
    """These are the main predicting settings. Set each before running
    this file."""

    if (len(sys.argv) == 5):
        seq_length = int(sys.argv[1])
        class_limit = int(sys.argv[2])
        model_path = sys.argv[3]
        dataset_path = sys.argv[4]
    else:
        print("Usage: python predict.py sequence_length class_limit model_path dataset_directory")
        print("Example: python predict.py 75 2 lstm-features.039-1.283.hdf5 /example_dir 720 1280")
        exit(1)

    sequences_dir = os.path.join('/content/drive/MyDrive/cnn/predict/data', 'sequences')
    if not os.path.exists(sequences_dir):
        os.mkdir(sequences_dir)

    checkpoints_dir = os.path.join('/content/drive/MyDrive/cnn/predict/data', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # model can be only 'lstm'
    model = 'lstm'
    saved_model = '/content/data/checkpoints/lstm-features.040-0.101.hdf5'  # None or weights file
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 1
    nb_epoch = 1
    data_type = 'features'
    image_shape = (image_height, image_width, 3)
    train_test = 'val'

    extract_features(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape, predict=True)
    predict(data_type, seq_length, model, video_path, saved_model=saved_model,
            class_limit=class_limit, image_shape=image_shape,
            load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch, train_test=train_test)


if __name__ == '__main__':
    main()
