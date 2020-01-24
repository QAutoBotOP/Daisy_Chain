from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras as keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Import Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

# Four NumPy arrays; train_images and train_labels are training set used to learn
# test_images and test_labels are what the model is tested against.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# The class names are not included with the dataset, so stored here:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Size of the dataset is: {}".format(train_images.shape))
print("Size of the testset is: {}".format(test_images.shape))

# Show first figure:
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# We see here that the pixel values fall in range of 0 to 255


# Set the scale of pixel values from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# To verify data is in right format, display first 25 img from training set, display class name below img.
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# Build the model
# 1. Configure the layers
# 2. Compile the model
model = keras.Sequential([
    # Flatten transform the format of the images from 2d array to 1d array
    keras.layers.Flatten(input_shape=(28, 28)),
    # Dense is fully connected layer, given nodes and activation function.
    # Softmax returns an array of 10 probability scores that sum to 1
    keras.layers.Dense(88, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Needs more setting;
# Loss function: Measures how accurate the model is during training, Minimize to steer in the right direction.
# Optimizer: How the model is updated based on the data it sees and its loss function.
# Metrics: Monitor the training and testing steps: accuracy fraction of images that are correctly classified.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# To start the training call model.fit - so called because it fits the model to the training data
# One run is called an epoch
model.fit(train_images, train_labels, epochs=10)

# Compare how the model performs on the test dataset:
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
# The accuracy on the test dataset is a little less than on the training dataset
# This gap represents overfitting; when a ML model performs worse on new, previously unseen inputs than on the data.
# An overfitted model memorizes the training data.

# With model trained, you can use it to make predictions about some images:
predictions = model.predict(test_images)
# Here, the model has predicted the label for each image in the testing set.
# Let's take a look at the first prediction:
print(predictions[0])
# Prediction is array of 10 numbers representing confidence that the image corresponds to each of the
# 10 different articles of clothing.
# To see the highest confidence value:
np.argmax(predictions[0])

# Examining the test label shows that this classification is correct:
print(test_labels[0])


# Graph to look at full set of predictions:
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color
               )


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Verify predictions: lets see 0th img.
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
#
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
# Note that a model can be wrong even when very confident.

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, predictions[i], test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

# Finally use the trained model to make a prediction about img
img = test_images[1]
print(img.shape)

# tf.keras models are optimized for batch predictions. So instead of single, create list:
# Add to batch where it is only member:
img = (np.expand_dims(img, 0))
print(img.shape)
# Now predict correct label for img:
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# model.predict returns a list of lists, one for each img in batch of data.
# Grab predictions for our only image in batch:
np.argmax(predictions_single[0])
