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
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import math

from utils import read_png, quantize_image, PackedTensors, write_png
from msh_model_test import build_model, SynthesisTransform, HyperSynthesisTransform

import matplotlib.pyplot as plt

CITYSCAPES_D1 = {
  "path":"/datatmp/Datasets/Cityscapes_extra/",
  "train_images_glob": "/datatmp/Datasets/Cityscapes_extra/leftImg8bit/train/*/*.png",
  "val_images_glob": "/datatmp/Datasets/Cityscapes_extra/leftImg8bit/val/*/*.png",
  "test_images_glob": "/datatmp/Datasets/Cityscapes_extra/leftImg8bit/test/*/*.png",
  "image_shape":[1024, 2048],
  "path_prefix":2,
}

IMAGE_PATH = "/datatmp/Datasets/Cityscapes_extra/leftImg8bit/val/wuppertal/wuppertal_000000_000016_leftImg8bit.png"

TRAINED_MODEL = "/datatmp/Experiments/comp_semseg/cityscapes/phase_1/msh/psnr/lambda_0.002/model.ckpt-200000"

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def compress(x, trained_model):
  _, _ , _, _, x_hat, y_hat, _, m, s, y, z, string, side_string, _ = \
    build_model(x, lmbda=0.002, mode='testing', msssim_loss=False)
  
  x_shape = tf.shape(x)[1:-1]
  y_shape = tf.shape(y)[1:-1]
  z_shape = tf.shape(z)[1:-1]
  tensors = [string, side_string, x_shape, y_shape, z_shape]  
  
  sess = tf.Session()
  tf.train.Saver().restore(sess, save_path=trained_model)  
  arrays, x_hat, m, s = sess.run([tensors, quantize_image(x_hat), m, s])
  packed = PackedTensors()
  packed.pack(tensors, arrays)
  return packed.string, x_hat, m, s
  
def decompress(packed_string, trained_model):
  string = tf.placeholder(tf.string, [1])
  side_string = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_shape = tf.placeholder(tf.int32, [2])
  z_shape = tf.placeholder(tf.int32, [2])  
  tensors = [string, side_string, x_shape, y_shape, z_shape]
  packed = PackedTensors(packed_string)
  arrays = packed.unpack(tensors)
  
  synthesis_transform = SynthesisTransform(192)
  hyper_synthesis_transform = HyperSynthesisTransform(192)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)

  z_shape = tf.concat([z_shape, [192]], axis=0)
  z_hat = entropy_bottleneck.decompress(side_string, z_shape, channels=192)
  mean, sigma = hyper_synthesis_transform(z_hat)
  mean = mean[:, :y_shape[0], :y_shape[1], :]
  sigma = sigma[:, :y_shape[0], :y_shape[1], :]
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(
      sigma, scale_table, mean=mean, dtype=tf.float32)
  y_hat = conditional_bottleneck.decompress(string)
  x_hat = synthesis_transform(y_hat)
  sess = tf.Session()
  tf.train.Saver().restore(sess, save_path=trained_model)  
  x_hat, m, s = sess.run([quantize_image(x_hat), mean, sigma], feed_dict=dict(zip(tensors, arrays)))
  return x_hat, m, s

def compress_decompress_test(dataset_constants, image_path, trained_model):
  def set_shape(x):
    x.set_shape(dataset_constants["image_shape"] + [3])
    return x

  x = tf.expand_dims(set_shape(read_png(image_path)), axis=0) 
  x_scaled = x / 255.

  string, bypassed_x_hat, bm, bs = compress(x_scaled, trained_model)
  tf.reset_default_graph()  
  actual_x_hat, m, s = decompress(string, trained_model)
  
  diff = np.abs(actual_x_hat - bypassed_x_hat)  
  fig, ax = plt.subplots(1, 3, figsize=(20.48*3, 10.24))  
  ax[0].imshow(bypassed_x_hat[0])
  ax[1].imshow(actual_x_hat[0])
  ax[2].imshow(diff[0])  
  fig.savefig("images")
  print(np.sum(diff))
  print(np.sum(np.abs(bm - m)))
  print(np.sum(np.abs(bs - s))) 

if __name__ == "__main__":
  compress_decompress_test(CITYSCAPES_D1, IMAGE_PATH, TRAINED_MODEL)
