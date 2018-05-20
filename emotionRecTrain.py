import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adamax
from keras.utils import np_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

parser = argparse.ArgumentParser()
# OPTIONAL
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--debug', dest='debug', action='store_true')

FLAGS = parser.parse_args()

if (FLAGS.debug):
    FLAGS.batch_size = 10
    FLAGS.n_epochs = 1

# todo argparse
raw_data_csv_file_name = 'data/fer2013.csv'
TRAIN_END = 28708
TEST_START = TRAIN_END + 1
IMG_SIZE = 48


def split_for_test(list):
    first = list[0:TRAIN_END]
    second = list[TEST_START:]
    return first, second


def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list


def process_emotion(emotion):
    emotion_as_list = pandas_vector_to_list(emotion)
    y_data = []
    for index in range(len(emotion_as_list)):
        y_data.append(emotion_as_list[index])

    # Y data
    y_data_categorical = np_utils.to_categorical(y_data, NUM_CLASSES)
    return y_data_categorical


def process_pixels(pixels, img_size=IMG_SIZE):
    pixels_as_list = pandas_vector_to_list(pixels)
    np_image_array = []
    for index, item in enumerate(pixels_as_list):
        data = np.zeros((img_size, img_size), dtype=np.uint8)
        pixel_data = item.split()
        for i in range(0, img_size):
            pixel_index = i * img_size
            data[i] = pixel_data[pixel_index:pixel_index + img_size]
        np_image_array.append(np.array(data))
    np_image_array = np.array(np_image_array)
    np_image_array = np_image_array.astype('float32') / 255.0
    return np_image_array


def get_vg_feature_map(vg, array_input, n_feature_maps):
    vg_input = duplicate_input_layer(array_input, n_feature_maps)

    picture_train_features = vg.predict(vg_input)
    del (vg_input)

    feature_map = np.empty([n_feature_maps, 512])
    for idx_pic, picture in enumerate(picture_train_features):
        feature_map[idx_pic] = picture
    return feature_map


def duplicate_input_layer(array_input, size):
    vg_input = np.empty([size, 48, 48, 3])
    for index, item in enumerate(vg_input):
        item[:, :, 0] = array_input[index]
        item[:, :, 1] = array_input[index]
        item[:, :, 2] = array_input[index]
    return vg_input


def main():
    K.set_learning_phase(0)
    raw_data = pd.read_csv(raw_data_csv_file_name)
    emotion_array = process_emotion(raw_data[['emotion']])
    pixel_array = process_pixels(raw_data[['pixels']])
    y_train, y_test = split_for_test(emotion_array)
    x_train_matrix, x_test_matrix = split_for_test(pixel_array)
    n_train = int(len(x_train_matrix))
    n_test = int(len(x_test_matrix))
    x_train_input = duplicate_input_layer(x_train_matrix, n_train)
    x_test_input = duplicate_input_layer(x_test_matrix, n_test)

    # vgg 16. include_top=False so the output is the 512
    vg = VGG16(include_top=False, input_shape=(48, 48, 3), weights='imagenet')

    # get feature map
    x_train_feature_map = get_vg_feature_map(vg, x_train_matrix, n_train)
    x_test_feature_map = get_vg_feature_map(vg, x_test_matrix, n_test)

    # very important to do this as a first thing

    # with tf.device('/gpu:0'):
    model = Sequential()
    model.add(Dense(256, input_shape=(512,), activation='relu'))
    model.add(Dense(128, input_shape=(256,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, input_shape=(64,)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    adamax = Adamax()

    model.compile(loss='categorical_crossentropy',
                  optimizer=adamax, metrics=['accuracy'])

    model.fit(x_train_feature_map, y_train,
              validation_data=(x_train_feature_map, y_train),
              nb_epoch=FLAGS.n_epochs, batch_size=FLAGS.batch_size)

    score = model.evaluate(x_test_feature_map,
                           y_test, batch_size=FLAGS.batch_size)

    print("Result: {}".format(score))

    inputs = Input(shape=(48, 48, 3))
    vg_output = vg(inputs)
    model_predictions = model(vg_output)
    final_model = Model(input=inputs, output=model_predictions)
    final_model.compile(loss='categorical_crossentropy',
                        optimizer=adamax, metrics=['accuracy'])
    final_model_score = final_model.evaluate(x_train_input,
                                             y_train, batch_size=FLAGS.batch_size)
    print("final model  train score: {}".format(final_model_score))

    final_model_score = final_model.evaluate(x_test_input,
                                             y_test, batch_size=FLAGS.batch_size)
    print("final model  test score: {}".format(final_model_score))

    config = final_model.get_config()
    weights = final_model.get_weights()
    new_model = Model.from_config(config)
    new_model.set_weights(weights)

    print("hi: ")
    print(new_model.input)

    export_path = 'fifHomeSavedModel'
    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'images': new_model.input},
                                      outputs={'scores': new_model.output})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()


if __name__ == "__main__":
    main()
