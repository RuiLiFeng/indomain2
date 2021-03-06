"""Perceptual module for encoder training."""
import tensorflow as tf

Model = tf.keras.models.Model
Flatten = tf.keras.layers.Flatten
Concatenate = tf.keras.layers.Concatenate
VGG16 = tf.keras.applications.vgg16.VGG16
preprocess_input = tf.keras.applications.vgg16.preprocess_input
import os
# from keras.models import Model
# from keras.layers import Flatten, Concatenate
# from keras.applications.vgg16 import VGG16, preprocess_input


class PerceptualModel(Model):
  """Defines the VGG16 model for perceptual loss."""

  def __init__(self, img_size, multi_layers=False):
    """Initializes with image size.

    Args:
      img_size: The image size prepared to feed to VGG16, default=256.
      multi_layers: Whether to use the multiple layers output of VGG16 or not.
    """
    super().__init__()
    if os.path.exists('/gdata2/fengrl/metrics/vgg.h5'):
      weights = '/gdata2/fengrl/metrics/vgg.h5'
    elif os.path.exists("/mnt/fengruili.fengruil/metrics/vgg.h5"):
      weights = "/mnt/fengruili.fengruil/metrics/vgg.h5"
    elif os.path.exists("/home/admin/workspace/project/data/metrics/vgg.h5"):
      weights = "/home/admin/workspace/project/data/metrics/vgg.h5"
    else:
      weights = 'imagenet'

    vgg = VGG16(include_top=False, input_shape=(img_size[0], img_size[1], 3), weights=weights)
    if multi_layers:
      layer_ids = [2, 5, 9, 13, 17]
      layer_outputs = [
          Flatten()(vgg.layers[layer_id].output) for layer_id in layer_ids]
      features = Concatenate(axis=-1)(layer_outputs)
    else:
      layer_ids = [13]  # 13 -> conv4_3
      features = [
          Flatten()(vgg.layers[layer_id].output) for layer_id in layer_ids]

    self._model = Model(inputs=vgg.input, outputs=features)

  def call(self, inputs, mask=None):
    return self._model(preprocess_input(inputs))

  def compute_output_shape(self, input_shape):
    return self._model.compute_output_shape(input_shape)
