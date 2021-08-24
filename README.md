# Padel video classification using InceptionV3 + LSTM:

Padel video classification using a pretrained InceptionV3 base model + a LSTM architecture.
This is a guide on how to execute it in Google Colab

1. Download the THETIS RGB dataset from <http://thetis.image.ece.ntua.gr/>
2. Upload the THETIS zipfile to Google Drive
3. Open the InceptionV3-LSTM.ipynb file is Google Collab
4. Unzip the dataset
6. Place the videos from the dataset in content/data/train and content/data/test folders. Each video type should have its own folder

>	| data/train
> >		| Forehand
> >		| Backhand
> >		...
>	| data/test
> >		| Forehand
> >		| Backhand
> >		...

7. Extract files from video with script extract_files.py. Pass video files extenssion as a param

`	$ python extract_files.py mp4`

8. Check the data_file.csv and choose the acceptable sequence length of frames. It should be less or equal to lowest one if you want to process all videos in dataset.
9. Extract sequence for each video with InceptionV3 and train LSTM. Run train.py script with sequence_length, class_limit, image_height, image_width args

`	$ python train.py 75 2 720 1280`

10. Save your best model file. (For example, lstm-features.hdf5)
11. Evaluate your model using predict.py. It will generate an .xlsx with the confusion matrix and the predictions for each video.

`	$ python train.py 75 2 720 1280`

12. Use clasify.py script to clasify your video. Args sequence_length, class_limit, saved_model_file, video_filename

`	$ python clasify.py 75 2 lstm-features.hdf5 video_file.mp4`

The result will be placed in result.avi file.

## Requirements

Ignore if you are using Google Colab

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

You must also have `ffmpeg` installed in order to extract the video files.

## Saved model

The weights of the model trained by us is too big to upload to Github. If you wish to use it contact us
