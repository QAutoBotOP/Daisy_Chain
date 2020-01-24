from __future__ import absolute_import, division, print_function, unicode_literals
import functools

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras as keras

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print(tf.__version__)

# Add training data:
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
# Add training file:
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read; precision is decimals, suppress is no 4*e-10
np.set_printoptions(precision=3, suppress=True)

# Load data; to start lets look at top of CSV file to see how it is formatted:
# Only one column needs to be identified explicitly. It is the one value
# that the model is intended to predict.
LABEL_COLUMN = 'survived'
LABELS = [0, 1]

# Now read CSV data from file and create dataset.
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs
    )
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

show_batch(raw_train_data)
# If file does not contain column names in first line, pass them in a list of strings
# to column_names argument in make_csv_dataset function.
# CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
# temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)



