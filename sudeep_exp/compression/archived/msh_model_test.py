"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_2")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True)),
        tfc.SignalConv2D(
            3, (5, 5), name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None)
    ]
    super(HyperAnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
        tensor = layer(tensor)
    return tensor

class HyperSynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=None),
        tfc.SignalConv2D(
            self.num_filters, (3, 3), name="layer_3", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=None)
    ]
    super(HyperSynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers[:-2]:
      tensor = layer(tensor)
    mean_layer = self._layers[-2]
    sigma_layer = self._layers[-1]
    return mean_layer(tensor), sigma_layer(tensor)


def build_model(x,
                lmbda,
                mode='training',
                layers=None,
                msssim_loss=False):
  """Builds the compression model."""  
  
  is_training = (mode == 'training')
  num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))

  if layers is None:
    num_filters = 192
    analysis_transform = AnalysisTransform(num_filters)
    synthesis_transform = SynthesisTransform(num_filters)
    hyper_analysis_transform = HyperAnalysisTransform(num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    
    layers = (analysis_transform, hyper_analysis_transform,
              entropy_bottleneck, hyper_synthesis_transform,
              synthesis_transform)
  else:
    analysis_transform, hyper_analysis_transform, entropy_bottleneck, \
    hyper_synthesis_transform, synthesis_transform = layers
  
  y = analysis_transform(x)
  z = hyper_analysis_transform(y)
  z_tilde_hat, z_likelihoods = entropy_bottleneck(z, training=is_training)
  mean, sigma = hyper_synthesis_transform(z_tilde_hat)
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, 
                                                   mean=mean)
  y_tilde_hat, y_likelihoods = conditional_bottleneck(y, training=is_training)
  x_tilde_hat = synthesis_transform(y_tilde_hat)

  if mode == "testing":
    side_string = entropy_bottleneck.compress(z_tilde_hat)
    string = conditional_bottleneck.compress(y_tilde_hat)  
  else:
    string = None
    side_string = None

  bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
          tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

  mse = tf.reduce_mean(tf.squared_difference(x, x_tilde_hat))
  mse *= 255 ** 2

  msssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(x_tilde_hat, x, 1))
  
  distortion = msssim if msssim_loss else mse 

  loss = lmbda * distortion + bpp
  
  return loss, bpp, mse, msssim, x_tilde_hat, y_tilde_hat, z_tilde_hat, \
         mean, sigma, y, z, string, side_string, layers


def decompress(compressed_file_path):
  string = tf.placeholder(tf.string, [1])
  side_string = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_shape = tf.placeholder(tf.int32, [2])
  z_shape = tf.placeholder(tf.int32, [2])
  with open(args.input_file, "rb") as f:
    packed = PackedTensors(f.read())
  tensors = [string, side_string, x_shape, y_shape, z_shape]
  arrays = packed.unpack(tensors)
