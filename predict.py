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

if (len(sys.argv) == 5):
    seq_length = int(sys.argv[1])
    class_limit = int(sys.argv[2])
    saved_model = sys.argv[3]
    video_path = sys.argv[4]
else:
    print ("Usage: python predict.py sequence_length class_limit saved_model_name video_directory")
    # TODO: example dir ?
    print ("Example: python predict.py 75 2 lstm-features.095-0.090.hdf5 /example_dir")
    exit (1)
