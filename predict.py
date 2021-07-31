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


def predict(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    # Helper: Save the model.
    # checkpointer = ModelCheckpoint(
    #     filepath=os.path.join('/content/drive/MyDrive/cnn/data', 'checkpoints', model + '-' + data_type + \
    #         '.{epoch:03d}-{val_loss:.3f}.hdf5'),
    #     verbose=1,
    #     save_best_only=True)
    #
    # # Helper: TensorBoard
    # tb = TensorBoard(log_dir=os.path.join('/content/drive/MyDrive/cnn/data', 'logs', model))
    #
    # # Helper: Stop when we stop learning.
    # early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    # timestamp = time.time()
    # csv_logger = CSVLogger(os.path.join('/content/drive/MyDrive/cnn/data', 'logs', model + '-' + 'training-' + \
    #     str(timestamp) + '.log'))

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
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        #print(*generator, sep='\n') # * will unpack the generator
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    #Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    generator_predict_train = data.frame_generator_predict(1, 'train', data_type)
    prediction_train = rm.model.predict_generator(generator_predict_train)
    print('Predictions:')
    print(prediction_train)
    #print('y')
    #print(y)
    #TODO: predicted labels

    # Format results and compute classification statistics
    dataset_name = 'THETIS'
    class_indices = {"backhand": 0, "forehand": 1}
    predicted_labels = np.argmax(prediction_train, axis=1).ravel().tolist()
    print('Predicted labels:')
    print(predicted_labels)
    results = Results(class_indices, dataset_name=dataset_name)

    #accuracy, confusion_matrix, classification = results.compute(test_generator.filenames, test_generator.classes,predicted_labels)
    # Display and save results
    #results.print(accuracy, confusion_matrix)

    if save:
        results.save(confusion_matrix, classification, predictions)

    # generator_predict_test = data.frame_generator_predict(1, 'test', data_type)
    # prediction_test = rm.model.predict_generator(generator_predict_test)
    # print('Predictions:')
    # print(prediction_test)


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
    saved_model = '/content/drive/MyDrive/cnn/model.h5'  # None or weights file
    load_to_memory = False # pre-load the sequences into memory
    batch_size = 1
    nb_epoch = 1
    data_type = 'features'
    image_shape = (image_height, image_width, 3)

    extract_features(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape, predict=True)
    predict(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
