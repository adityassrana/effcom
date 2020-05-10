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

import glob
import sys
import os
import time 
import cv2

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib.framework import get_variables_to_restore

import tensorflow_compression as tfc
from .utils import read_png, quantize_image
from .msh_model import build_model

def log_all_summaries(unscaled_x, x_tilde, loss, bpp, mse, msssim, mode):
    summaries = []    
    summaries += [tf.summary.image(mode+'_input_image', unscaled_x)]
    summaries += [tf.summary.scalar(mode+'_loss', loss)]
    summaries += [tf.summary.scalar(mode+"_bpp", bpp)]
    summaries += [tf.summary.scalar(mode+"_mse", mse)]
    summaries += [tf.summary.image(mode+"_reconstruction",
                  quantize_image(x_tilde))]
    summaries += [tf.summary.scalar(mode+"_msssim_dB",
                  -10*tf.log(msssim) / np.log(10))] 
    summary = tf.summary.merge(summaries)    
    return summary 

def fit(dataset_constants,
        lmbda,
        pretrained_model,
        checkpoint_dir,
        msssim_loss=False,
        preprocess_threads=8,
        patchsize=256,
        batchsize=8,
        last_step=200000,
        validate=False):
  """Trains the model."""
  
  train_glob = dataset_constants["train_images_glob"]
  x_train_files = sorted(glob.glob(train_glob))
  
  train_dataset = tf.data.Dataset.from_tensor_slices(x_train_files)

  train_dataset = train_dataset.shuffle(buffer_size=len(x_train_files)).repeat()
  train_dataset = train_dataset.map(read_png,
                                    num_parallel_calls=preprocess_threads)
  train_dataset = train_dataset.map(lambda x: tf.random_crop(x,
                                                (patchsize, patchsize, 3)))
  train_dataset = train_dataset.batch(batchsize)
  train_dataset = train_dataset.prefetch(batchsize)

  x_train_unscaled = train_dataset.make_one_shot_iterator().get_next()
  x_train = x_train_unscaled / 255.  
  
  train_loss, train_bpp, train_mse, train_msssim, x_tilde, \
    _, _, _, _, _, _, layers = build_model(x_train, lmbda,
                                     mode = 'training', msssim_loss=msssim_loss)  
  train_summary = log_all_summaries(x_train_unscaled, x_tilde, train_loss,
                                    train_bpp, train_mse, train_msssim, "train")
  
  if validate:
    def set_shape(x):
      x.set_shape(dataset_constants["image_shape"] +[3])
      return x

    val_batchsize = batchsize // 4
    val_preprocess_threads = min(preprocess_threads, val_batchsize)
    val_glob = dataset_constants["val_images_glob"]
    x_val_files = glob.glob(val_glob)
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val_files)
    val_dataset = val_dataset.shuffle(buffer_size=len(x_val_files)).repeat()
    val_dataset = val_dataset.map(read_png, 
                                  num_parallel_calls=val_preprocess_threads)
    val_dataset = val_dataset.map(set_shape,
                                  num_parallel_calls=val_preprocess_threads)
    val_dataset = val_dataset.batch(val_batchsize)
    val_dataset = val_dataset.prefetch(val_batchsize)
    x_val_unscaled = val_dataset.make_one_shot_iterator().get_next()    
    x_val = x_val_unscaled / 255.

    val_loss, val_bpp, val_mse, val_msssim, x_hat, _, _, _, _,_, _, _  = \
      build_model(x_val, lmbda, 'testing', layers, msssim_loss)
    val_summary = log_all_summaries(x_val_unscaled, x_hat, val_loss, val_bpp,
                                    val_mse, val_msssim, "val") 

  step = tf.train.get_or_create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  _, _, entropy_bottleneck, _, _ = layers  

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(sum(entropy_bottleneck.losses))
  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates)

  hooks = [
    tf.train.StopAtStepHook(last_step=last_step),
    tf.train.NanTensorHook(train_loss),
    tf.train.SummarySaverHook(save_secs=120, output_dir=checkpoint_dir,
                                summary_op=train_summary)
  ] 
  if validate:
    hooks += [
      tf.train.SummarySaverHook(save_secs=120, output_dir=checkpoint_dir,
                                 summary_op=val_summary)
    ]
  exclude_list = ['global_step']
  variables_to_restore = get_variables_to_restore(exclude=exclude_list)
  pre_train_saver = tf.train.Saver(variables_to_restore)

  def load_pretrain(scaffold, sess):
    pre_train_saver.restore(sess, save_path=pretrained_model)

  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=checkpoint_dir,
      save_checkpoint_steps=25000, save_summaries_secs=None,
      save_summaries_steps=None, 
      scaffold=tf.train.Scaffold(init_fn=load_pretrain,
                                saver=tf.train.Saver(max_to_keep=11))) as sess:
    while not sess.should_stop():
      sess.run(train_op)
