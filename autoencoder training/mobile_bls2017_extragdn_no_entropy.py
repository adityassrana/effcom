# Lint as: python3
# Copyright 2018 Google LLC. All Rights Reserved.
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
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

import argparse
import glob
import sys
import os
import json

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc


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
  """The faster analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            1, (5, 5), name="layer_1dw", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            channel_separable=True,
            activation=tfc.GDN(name="gdn_1dw")),
        tfc.SignalConv2D(
            self.num_filters, (1, 1), name="layer_1pw", corr=True, strides_down=1,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1pw")),
        tfc.SignalConv2D(
            1, (5, 5), name="layer_2dw", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            channel_separable=True,
            activation=tfc.GDN(name="gdn_2dw")),
        tfc.SignalConv2D(
            self.num_filters, (1, 1), name="layer_2pw", corr=True, strides_down=1,
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
            1, (5, 5), name="layer_0dw", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0dw", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (1, 1), name="layer_0pw", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            1, (5, 5), name="layer_1dw", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            channel_separable=True,
            activation=tfc.GDN(name="igdn_1dw", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (1, 1), name="layer_1pw", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        tfc.SignalConv2D(
            3, (9, 9), name="layer_3", corr=False, strides_up=4,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


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

  #num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  #entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Build autoencoder.
  y = analysis_transform(x)
  #y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y)

  # Total number of bits divided by number of pixels.
  #train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))

  # Multiply by 255^2 to correct for rescaling.
  #train_mse *= 255 ** 2

  # Calculate psnr and ssim
  train_psnr = tf.reduce_mean(tf.image.psnr(x_tilde, x, 1))
  train_msssim_value = tf.reduce_mean(tf.image.ssim_multiscale(x_tilde, x, 1))

  # structural similarity loss
  train_ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(x_tilde, x, 1))

  #Choose distortion metric
  distortion = train_ssim if args.ssim_loss else train_mse
  
  # The rate-distortion cost.
  train_loss = distortion

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  #aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  #aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step)

  # Log scalar values
  s_loss = tf.summary.scalar("train/loss", train_loss)
  #s_bpp = tf.summary.scalar("train/bpp", train_bpp)
  s_mse = tf.summary.scalar("train/mse", train_mse)
  s_psnr = tf.summary.scalar("train/psnr", train_psnr)
  s_msssim_value = tf.summary.scalar("train/multiscale ssim value", train_msssim_value)
  s_ssim = tf.summary.scalar("train/multiscale ssim", -10 * tf.log(train_ssim)) 

  # Log training images
  s_original = tf.summary.image("images/original", quantize_image(x))
  s_reconstruction = tf.summary.image("images/reconstruction", quantize_image(x_tilde))

  # Merge scalars into a summary
  train_summary = tf.summary.merge([s_loss, s_mse, s_psnr, s_msssim_value, s_ssim])

  #Merge images into a summary
  image_summary = tf.summary.merge([s_original, s_reconstruction])

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
      tf.train.SummarySaverHook(save_secs=30,output_dir=args.checkpoint_dir,summary_op=train_summary),
      tf.train.SummarySaverHook(save_secs=3600,output_dir=args.checkpoint_dir,summary_op=image_summary)
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_steps=None, save_summaries_secs=None) as sess:
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

  # Load input image and add batch dimension.
  x = read_png(args.input_file)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])
  x_shape = tf.shape(x)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Transform and compress the image.
  y = analysis_transform(x)
  string = entropy_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)
  x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

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
    tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
    arrays = sess.run(tensors)

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    packed.pack(tensors, arrays)
    with open(args.output_file, "wb") as f:
      f.write(packed.string)

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
    #return num_pixels, bpp, eval_bpp, mse, psnr, msssim


def decompress(args):
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  string = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_shape = tf.placeholder(tf.int32, [2])
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = [string, x_shape, y_shape]
  arrays = packed.unpack(tensors)

  # Instantiate model.
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Decompress and transform the image back.
  y_shape = tf.concat([y_shape, [args.num_filters]], axis=0)
  y_hat = entropy_bottleneck.decompress(
      string, y_shape, channels=args.num_filters)
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

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=64,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--gpu", type=str, default='4',
      help="Which gpu to use")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="/datatmp/Datasets/CLIC_2019_Professional/CLIC_2019_Images/train/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--valid_glob", default="./valid_high/*.png",
      help="Glob pattern identifying testing data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--test_glob", default="/datatmp/Datasets/CLIC_2019_Professional/CLIC_2019_Images/valid/*.png",
      help="Glob pattern identifying testing data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--kodak_glob", default="./kodak/*.png",
      help="Glob pattern identifying testing data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--experiment_name", default="64filters_depthwise_extraGDN_noentropy",
      help="Directory where to save/load model checkpoints.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.002,dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=800000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=12,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--valid_patchsize", type=int, default=256)


  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))


  # Sudeep's arguments
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
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args

def experiment(args):
  if args.command == "train":
    exp_dir = os.path.join("/datatmp/Experiments/aditya", args.experiment_name)
    args.checkpoint_dir = exp_dir
    try:
      os.mkdir(args.checkpoint_dir)
    except OSError:
      print("Directory %s already exists" % args.checkpoint_dir)
    else:
      print ("Creating experiment directory %s " % args.checkpoint_dir)

    with open(os.path.join(exp_dir,'args1.txt'), 'w') as f:
      json.dump(args.__dict__, f, indent=2)
      print('Dump argsparse text file')
    print('Start Training')
    train(args)

  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)

def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(experiment, flags_parser=parse_args)
