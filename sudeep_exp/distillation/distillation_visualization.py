from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import os
import time 
import cv2
import pickle

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import tensorflow_compression as tfc

from logging_formatter import Logger
logger = Logger()

from deeplab import model
from deeplab import common

PATH_TO_TRAINED_MODEL = '/datatmp/Experiments/semantic_compression/cityscapes/mobilenetv2_trained/deeplabv3_mnv2_cityscapes_train/model.ckpt'

tf.random.set_random_seed(12)

flags = tf.app.flags
FLAGS = flags.FLAGS


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image

def read_pngs(filename1, label_filename1, filename2):
  """Loads a PNG image file."""
  string = tf.read_file(filename1)
  image1 = tf.image.decode_image(string, channels=3)
  string = tf.read_file(label_filename1)
  label_image1 = tf.image.decode_image(string, channels=1)
  string = tf.read_file(filename2)
  image2 = tf.image.decode_image(string, channels=3)
  image = tf.cast(tf.concat([image1, label_image1, image2], axis=-1), tf.float32)
  return image

def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

def experiment(l_args):
  x_val_files = sorted(glob.glob('/datatmp/Datasets/Cityscapes/leftImg8bit/val/*/*.png'))
  x_label_files = sorted(glob.glob('/datatmp/Datasets/Cityscapes/gtFine/val/*/*_labelIds.png'))
  y_val_files = sorted(glob.glob('/datatmp/Experiments/semantic_compression/{}/lambda_{}/leftImg8bit/val/*/*.png'.format(l_args.images_dir, l_args.lmbda)))

  print(len(x_val_files), len(y_val_files))  
  assert(len(x_val_files) == len(y_val_files))
  assert(x_val_files[0].split("/")[-1] == y_val_files[0].split("/")[-1])  
  assert(x_val_files[-1].split("/")[-1] == y_val_files[-1].split("/")[-1])    
  
  print(x_val_files[0].split("/")[-1][:-16], x_label_files[0].split("/")[-1].split("_gtFine_labelIds.png")[0])
  print(x_val_files[-1].split("/")[-1][:-16], x_label_files[-1].split("/")[-1].split("_gtFine_labelIds.png")[0])
  assert(len(x_label_files) == len(x_val_files))
  assert(x_val_files[0].split("/")[-1][:-16] == x_label_files[0].split("/")[-1].split("_gtFine_labelIds.png")[0])  
  assert(x_val_files[-1].split("/")[-1][:-16] == x_label_files[-1].split("/")[-1].split("_gtFine_labelIds.png")[0])    

  def set_shape(x):
      x.set_shape([1024, 2048, 7])
      return x

  val_dataset = tf.data.Dataset.from_tensor_slices((x_val_files,x_label_files, y_val_files))
  val_dataset = val_dataset.map(read_pngs, num_parallel_calls=l_args.preprocess_threads)
  val_dataset = val_dataset.map(set_shape, num_parallel_calls=l_args.preprocess_threads)
  val_dataset = val_dataset.batch(1)
  val_dataset = val_dataset.prefetch(1)
  val_batch = val_dataset.make_one_shot_iterator().get_next()

  val_x, _, val_y = val_batch[:, :, :, :3] , val_batch[:, :, :, 3:4], val_batch[:, :, :, 4:]
  scaled_val_x, scaled_val_y = val_x / 255., val_y / 255.

  model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: 19},
        crop_size=[int(z) for z in l_args.patchsize.split(",")],
        atrous_rates=None,
        output_stride=16)

  x_features, _ = model.extract_features(val_x, model_options)
  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    y_features, _ = model.extract_features(val_y, model_options)

  exclude_list = ['global_step']
  variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude_list)
  seg_saver = tf.train.Saver(variables_to_restore)
  
  diff_features = x_features -  y_features

  sess = tf.Session()
  seg_saver.restore(sess, save_path=PATH_TO_TRAINED_MODEL)

  #while not sess.should_stop():
  feature_index = 256      
  for i in range(500):
    x, y, fx, fy, dxy = sess.run([val_x, val_y, x_features, y_features, diff_features])
    print(i)    
    rxy = np.maximum(fx, 1e-6) / np.maximum(fy, 1e-6)
    #for i in range(fx.shape[-1]):  
    fig = plt.figure(figsize=(30, 20))
    ax1 = fig.add_subplot(3,2,1)
    ax1.imshow(x[0].astype(np.uint8))
    ax2 = fig.add_subplot(3,2,2)
    ax2.imshow(y[0].astype(np.uint8))
    ax3 = fig.add_subplot(3,2,3)
    ax3.imshow(fx[0, :, :, feature_index]/np.max(fx[0, :, :, feature_index]))
    ax4 = fig.add_subplot(3,2,4)
    ax4.imshow(fy[0, :, :, feature_index]/np.max(fy[0, :, :, feature_index]))
    ax5 = fig.add_subplot(3,2,5)
    ax5.imshow(np.abs(dxy[0, :, :,  feature_index])/np.max(np.abs(dxy[0, :, :,  feature_index])))
    ax6 = fig.add_subplot(3,2,6)
    ax6.imshow(rxy[0, :, :,  feature_index]/np.max(rxy[0, :, :,  feature_index]))
    plt.savefig("feats/{}.png".format(i))
    

def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--batchsize", type=int, default=4,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=str, default="256,256",
      help="Size of image patches for training.")
  parser.add_argument(
      "--lr", type=float, default=0.001)
  parser.add_argument(
      "--mu", type=float, default=0.5)
  parser.add_argument(
      "--lmbda", type=float, default=0.002,
      help="list of lambda values that the model will be trained with")
  parser.add_argument(
      "--exp-name", default="restoration_distillation",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--exp-base", default="/datatmp/Experiments/semantic_compression",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--images-dir", default="recon_new_msh_ft_cityscapes",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--last_step", type=int, default=50000,
      help="Train up to this number of steps.")
  parser.add_argument(
      "--preprocess_threads", type=int, default=8,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  parser.add_argument(
      "--valid_patchsize", type=int, default=512)
  parser.add_argument(
      "--gpu", type=str, default='0',
      help="Gpu to use")
  parser.add_argument("--loss_type", type=str, default="msssim")   
  parser.add_argument('--test_only', dest='test_only', action='store_true')
  
  # Parse arguments.  
  args = parser.parse_args(argv[1:])
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  return args

def main(unused_argv):
  FLAGS.aspp_with_batch_norm = False
  FLAGS.aspp_with_separable_conv = False
  FLAGS.decoder_use_separable_conv = False
  app.run(experiment, flags_parser=parse_args)

if __name__ == "__main__":
  tf.app.run()
