# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

import argparse
import glob
import sys
import os
import time 
import cv2

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf

import tensorflow_compression as tfc


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

# TODO(jonycgn): Use tfc.PackedTensors once new binary packages have been built.
class PackedTensors(object):
  """Packed representation of compressed tensors."""

  def __init__(self, string=None):
    self._example = tf.train.Example()
    if string:
      self.string = string

  @property
  def model(self):
    """Model identifier."""
    buf = self._example.features.feature["MD"].bytes_list.value[0]
    return buf.decode("ascii")

  @model.setter
  def model(self, value):
    self._example.features.feature["MD"].bytes_list.value[:] = [
        value.encode("ascii")]

  @model.deleter
  def model(self):
    del self._example.features.feature["MD"]

  @property
  def string(self):
    """A string representation of this object."""
    return self._example.SerializeToString()

  @string.setter
  def string(self, value):
    self._example.ParseFromString(value)

  def pack(self, tensors, arrays):
    """Packs Tensor values into this object."""
    if len(tensors) != len(arrays):
      raise ValueError("`tensors` and `arrays` must have same length.")
    i = 1
    for tensor, array in zip(tensors, arrays):
      feature = self._example.features.feature[chr(i)]
      feature.Clear()
      if array.ndim != 1:
        raise RuntimeError("Unexpected tensor rank: {}.".format(array.ndim))
      if tensor.dtype.is_integer:
        feature.int64_list.value[:] = array
      elif tensor.dtype == tf.string:
        feature.bytes_list.value[:] = array
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))
      i += 1
    # Delete any remaining, previously set arrays.
    while chr(i) in self._example.features.feature:
      del self._example.features.feature[chr(i)]
      i += 1

  def unpack(self, tensors):
    """Unpacks Tensor values from this object."""
    arrays = []
    for i, tensor in enumerate(tensors):
      feature = self._example.features.feature[chr(i + 1)]
      np_dtype = tensor.dtype.as_numpy_dtype
      if tensor.dtype.is_integer:
        arrays.append(np.array(feature.int64_list.value, dtype=np_dtype))
      elif tensor.dtype == tf.string:
        arrays.append(np.array(feature.bytes_list.value, dtype=np_dtype))
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))
    return arrays


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


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











def train(args):
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  synthesis_transform = SynthesisTransform(args.num_filters)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()

  # Build autoencoder and hyperprior.
  y = analysis_transform(x)
  z = hyper_analysis_transform(y)
  z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
  mean, sigma = hyper_synthesis_transform(z_tilde)
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mean)
  y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde)

  # Total number of bits divided by number of pixels.
  train_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
               tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2
  
  train_ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(x_tilde, x, 1))
  
  distortion = train_ssim if args.ssim_loss else train_mse 
  # The rate-distortion cost.
  train_loss = args.lmbda * distortion + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(sum(entropy_bottleneck.losses))
  
  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates)

  s_train_loss = tf.summary.scalar("train_loss", train_loss)
  s_bpp = tf.summary.scalar("bpp", train_bpp)
  s_mse = tf.summary.scalar("mse", train_mse)
  s_ssim = tf.summary.scalar("multiscale ssim", -10 * tf.log(train_ssim) / np.log(10))
  s_ylikelihoods = tf.summary.scalar("y_likelihoods", tf.reduce_sum(y_likelihoods))
  s_zlikelihoods = tf.summary.scalar("z_likelihoods", tf.reduce_sum(z_likelihoods))
  train_summary = tf.summary.merge([s_train_loss, s_bpp, s_mse, s_ylikelihoods, s_zlikelihoods, s_ssim])
  
  
  #### Validation ####
  def generic_central_crop(size):    
    def _crop(image, size):      
      h, w = tf.shape(image)[0], tf.shape(image)[1]
      image_crop = image[h//2 - size//2: h//2 + size//2, w//2 - size//2 :w//2 + size//2]
      image_crop.set_shape([512, 512, 3])
      return image_crop
    return lambda x: _crop(x, size)

  central_crop = generic_central_crop(args.valid_patchsize)

  with tf.device("/cpu:0"):
    valid_files = glob.glob(args.valid_glob)
    if not valid_files:
      raise RuntimeError(
          "No validation images found with glob '{}'.".format(args.valid_glob))
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_files)
    valid_dataset = valid_dataset.repeat()
    valid_dataset = valid_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    valid_dataset = valid_dataset.map(lambda x: central_crop(x))
    valid_dataset = valid_dataset.batch(args.batchsize)
    valid_dataset = valid_dataset.prefetch(2*args.batchsize)
  
  valid_x = valid_dataset.make_one_shot_iterator().get_next()
 
  # Build autoencoder and hyperprior.
  valid_y = analysis_transform(valid_x)
  valid_z = hyper_analysis_transform(valid_y)
  valid_z_hat, valid_z_likelihoods = entropy_bottleneck(valid_z, training=False)
  valid_mean, valid_sigma = hyper_synthesis_transform(valid_z_hat)
  valid_conditional_bottleneck = tfc.GaussianConditional(valid_sigma, scale_table, mean=valid_mean)
  valid_y_hat, valid_y_likelihoods = valid_conditional_bottleneck(valid_y, training=False)
  valid_x_hat = synthesis_transform(valid_y_hat)
  
  valid_bpp = (tf.reduce_sum(tf.log(valid_y_likelihoods)) +
               tf.reduce_sum(tf.log(valid_z_likelihoods))) / (-np.log(2) * (args.batchsize * args.valid_patchsize ** 2))
  
  valid_mse = tf.reduce_mean(tf.squared_difference(valid_x, valid_x_hat))
  valid_ssim = tf.reduce_mean(-10 * tf.log(1 - tf.image.ssim_multiscale(valid_x_hat, valid_x, 1)) / np.log(10))
  valid_distortion = valid_ssim if args.ssim_loss else valid_mse 
  valid_mse *= 255 ** 2
  valid_loss = args.lmbda * valid_distortion + valid_bpp
  
  s_original = tf.summary.image("valid_original", quantize_image(valid_x), max_outputs=3)
  s_reconstruction = tf.summary.image("valid_reconstruction", quantize_image(valid_x_hat), max_outputs=3)
  s_valid_loss = tf.summary.scalar("valid_loss", valid_loss)
  s_valid_mse = tf.summary.scalar("valid_mse", valid_mse)
  s_valid_ssim = tf.summary.scalar("valid_msssim", valid_ssim)
  s_valid_bpp = tf.summary.scalar("valid_bpp", valid_bpp)
  valid_summary = tf.summary.merge([s_valid_loss, s_valid_mse, s_valid_bpp, s_original, s_reconstruction, s_valid_ssim])
  
  veeterzy_index = 5
  veeterzy_sigma = valid_sigma[veeterzy_index]
  veeterzy_mean = valid_mean[veeterzy_index]
  veeterzy_latents = valid_y_hat[veeterzy_index]
  veertezy_likelihoods = valid_y_likelihoods[veeterzy_index]
  channel_index = tf.argmin(tf.reduce_sum(veertezy_likelihoods, axis=[0, 1]))
  feature = veeterzy_latents[:, :, channel_index]
  feature_mean = veeterzy_mean[:, :, channel_index]
  feature_sigma = veeterzy_sigma[:, :, channel_index]
  
  feature_normalized = (feature - feature_mean) / feature_sigma
  s_feature_n = tf.summary.image("veeterzy-82537_normalized", tf.reshape(feature_normalized, [-1, feature.shape[0], feature.shape[1], 1]))
  s_feature = tf.summary.image("veeterzy-82537", tf.reshape(feature, [-1, feature.shape[0], feature.shape[1], 1]))
  features_summary = tf.summary.merge([s_feature_n, s_feature])
  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
      tf.train.SummarySaverHook(save_secs=60,output_dir=args.checkpoint_dir,summary_op=train_summary),
      tf.train.SummarySaverHook(save_secs=120,output_dir=args.checkpoint_dir,summary_op=valid_summary),
      tf.train.SummarySaverHook(save_secs=300,output_dir=args.checkpoint_dir,summary_op=features_summary)
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=600, save_summaries_steps=None, save_summaries_secs=None) as sess:
    while not sess.should_stop():
      sess.run(train_op)

def test(args):
  test_files = glob.glob(args.test_glob)
  with open(args.results_file, "w+") as f:
    f.write("{},{},{},{},{},{},{}\n".format("test_file", "num_pixels", "bpp", "eval_bpp", "mse", "psnr", "msssim"))
    for test_file in test_files:
      tf.reset_default_graph()
      num_pixels, bpp, eval_bpp, mse, psnr, msssim = compress(args, test_file)
      with open(args.results_file, "a") as f:
        f.write("{},{},{},{},{},{},{}\n".format(test_file, num_pixels, bpp, eval_bpp, mse, psnr, msssim))

def compress(args, image_path=None):
  """Compresses an image."""
  if image_path is None:
    x = load_image(args.input)
  else:
    x = read_png(image_path)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  synthesis_transform = SynthesisTransform(args.num_filters)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()

  # Transform and compress the image.
  y = analysis_transform(x)
  y_shape = tf.shape(y)
  z = hyper_analysis_transform(y)
  z_hat, z_likelihoods = entropy_bottleneck(z, training=False)
  mean, sigma = hyper_synthesis_transform(z_hat)
  mean, sigma = mean[:, :y_shape[1], :y_shape[2], :], sigma[:, :y_shape[1], :y_shape[2], :]

  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mean)
  side_string = entropy_bottleneck.compress(z)
  string = conditional_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)[:, :tf.shape(x)[1], :tf.shape(x)[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
              tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    tensors = [string, side_string,
               tf.shape(x)[1:-1], tf.shape(y)[1:-1], tf.shape(z)[1:-1]]
    arrays = sess.run(tensors)

    # Write a binary file with the shape information and the compressed string.
    packed = PackedTensors()
    packed.pack(tensors, arrays)
    #with open(args.output_file, "wb") as f:
    #  f.write(packed.string)

    # If requested, transform the quantized image back and measure performance.
    eval_bpp, mse, psnr, msssim, num_pixels = sess.run([eval_bpp, mse, psnr, msssim, num_pixels])

    # The actual bits per pixel including overhead.
    bpp = len(packed.string) * 8 / num_pixels

    print("Mean squared error: {:0.4f}".format(mse))
    print("PSNR (dB): {:0.2f}".format(psnr))
    print("Multiscale SSIM: {:0.4f}".format(msssim))
    print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
    print("Information content in bpp: {:0.4f}".format(eval_bpp))
    print("Actual bits per pixel: {:0.4f}".format(bpp))
    return num_pixels, bpp, eval_bpp, mse, psnr, msssim

def decompress(args):
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  string = tf.placeholder(tf.string, [1])
  side_string = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_shape = tf.placeholder(tf.int32, [2])
  z_shape = tf.placeholder(tf.int32, [2])
  with open(args.input_file, "rb") as f:
    packed = PackedTensors(f.read())
  tensors = [string, side_string, x_shape, y_shape, z_shape]
  arrays = packed.unpack(tensors)

  # Instantiate model.
  synthesis_transform = SynthesisTransform(args.num_filters)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)

  # Decompress and transform the image back.
  z_shape = tf.concat([z_shape, [args.num_filters]], axis=0)
  z_hat = entropy_bottleneck.decompress(
      side_string, z_shape, channels=args.num_filters)
  sigma = hyper_synthesis_transform(z_hat)
  sigma = sigma[:, :y_shape[0], :y_shape[1], :]
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(
      sigma, scale_table, dtype=tf.float32)
  y_hat = conditional_bottleneck.decompress(string)
  x_hat = synthesis_transform(y_hat)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = write_png(args.output_file, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op, feed_dict=dict(zip(tensors, arrays)))



def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters per layer.")
  parser.add_argument(
      "--experiment_name", default="train",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--train_glob", default='/datatmp/Datasets/CLIC_2019_Professional/CLIC_2019_Images/train/*.png',
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  parser.add_argument(
      "--valid_glob", default="./valid_high/*.png",
      help="Glob pattern identifying testing data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  parser.add_argument(
      "--test_glob", default="/datatmp/Datasets/CLIC_2019_Professional/CLIC_2019_Images/valid/*.png",
      help="Glob pattern identifying testing data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  parser.add_argument(
      "--kodak_glob", default="./kodak/*.png",
      help="Glob pattern identifying testing data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  parser.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambdas", type=str, default='0.001,0.002,0.004,0.008,0.016,0.032,0.064',
      help="list of lambda values that the model will be trained with")
  parser.add_argument(
      "--last_step", type=int, default=600000,
      help="Train up to this number of steps.")
  parser.add_argument(
      "--preprocess_threads", type=int, default=12,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  parser.add_argument(
      "--valid_patchsize", type=int, default=512)
  parser.add_argument(
      "--gpu", type=str, default='1',
      help="Gpu to use")
  parser.add_argument('--ssim', dest='ssim_loss', action='store_true')
  parser.set_defaults(ssim_loss=False)    
  parser.add_argument('--test_only', dest='test_only', action='store_true')
  parser.set_defaults(test_only=False)
  parser.add_argument('--save_compressed_reconstructed', dest='save', action='store_true')
  parser.set_defaults(save=False)
  parser.add_argument('--kodak', dest='kodak', action='store_true')
  parser.set_defaults(kodak=False)

  # Parse arguments.  
  args = parser.parse_args(argv[1:])
  args.test_only = args.test_only or args.kodak
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  return args

def experiment(args):  
  exp_dir = os.path.join("/datatmp/Experiments/skatakol", args.experiment_name)
  if args.ssim_loss:
    exp_dir += "_ms-ssim"
  if not args.test_only:
    if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
      print('Creating experiment directory ' + exp_dir + '/')
    else:
      print('Experiment directory already exist')

  lambdas = args.lambdas.split(',')
  lambdas = [float(x) for x in lambdas]
  
  for lmbda in lambdas:
    train_dir = os.path.join(exp_dir, 'lambda_'+str(lmbda))
    train_time_path = os.path.join(train_dir, 'time_analysis.txt')
    args.checkpoint_dir = train_dir
    args.lmbda = lmbda
    train_time_secs, test_time_secs = 0, 0
    if not args.test_only:
      if not os.path.exists(train_dir):
        print('Creating subdir in experiment directory for lambda = '+str(lmbda))
        os.makedirs(train_dir)
      try:
        print('Start training')
        train_time_st = time.time()
    
        train(args)

        tf.reset_default_graph()

        train_time_secs = int(time.time() - train_time_st)
      
      except Exception as e:
        print("Error {}".format(str(e)))
        #shutil.rmtree(train_dir)
        raise 

    if not os.path.exists(train_dir):
      continue  
    
    if args.kodak:
      kodak_dir = os.path.join(exp_dir, "kodak")
      if not os.path.exists(kodak_dir):
        os.makedirs(kodak_dir)
        print('Creating kodak dir')     
      args.results_file = os.path.join(kodak_dir, 'lambda_'+str(lmbda)+".csv")
      args.test_glob = args.kodak_glob
      if not os.path.exists(args.results_file):
        print('Creating result file for lambda = '+str(lmbda))
        try:
          print('Start testing')
          test_time_st = time.time()
          test(args)
          tf.reset_default_graph()
          test_time_secs = int(time.time() - test_time_st)
        except Exception as e:
          print("Error {}".format(str(e)))
          os.remove(args.results_file)
          raise
      else:
        print('Tested with lambda= '+str(lmbda)+' before, skipping')
    else:    
      args.results_file = os.path.join(exp_dir, 'lambda_'+str(lmbda)+".csv")
      if not os.path.exists(args.results_file):
        print('Creating result file for lambda = '+str(lmbda))
        try:
          print('Start testing')
          test_time_st = time.time()
          test(args)
          tf.reset_default_graph()
          test_time_secs = int(time.time() - test_time_st)
        except Exception as e:
          print("Error {}".format(str(e)))
          os.remove(args.results_file)
          raise
      else:
        print('Tested with lambda= '+str(lmbda)+' before, skipping')

if __name__ == "__main__":
  app.run(experiment, flags_parser=parse_args)