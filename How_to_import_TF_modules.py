import tensorflow as tf
import keras as ker
import tensorflow_hub as hub

module = hub.module("https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4")

features = module(my_images)
logits = tf.layers.dense(features, NUM_CLASSES)
probabilities = tf.nn.softmax(logits)



