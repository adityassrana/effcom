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
import cv2
import math 
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
import math
from .utils import read_png, quantize_image, PackedTensors, write_png
from .msh_model import build_model

def evaluation(dataset_constants,
               lmbda,
               trained_model,
               save_dir,
               save_images=True,
               split="test",
               msssim_loss=False):
  
  print("Attempting evaluation on {}".format(trained_model))

  compressed_reconstructed_dir = save_dir
  metrics_path = os.path.join(save_dir, 
           "{}_{}_average_metrics.pkl".format(dataset_constants["name"], split))
  def set_shape(x):
    x.set_shape(dataset_constants["image_shape"] + [3])
    return x
  
  val_batchsize = 1
  val_preprocess_threads = 1
  test_glob = dataset_constants["{}_images_glob".format(split)]
  x_val_files = glob.glob(test_glob)
  val_dataset = tf.data.Dataset.from_tensor_slices(x_val_files)
  val_dataset = val_dataset.map(read_png, 
                                num_parallel_calls=val_preprocess_threads)
  val_dataset = val_dataset.map(set_shape,
                                num_parallel_calls=val_preprocess_threads)
  val_dataset = val_dataset.batch(val_batchsize)
  val_dataset = val_dataset.prefetch(val_batchsize)
  x = val_dataset.make_one_shot_iterator().get_next()    
  x_scaled = x / 255.

  loss, eval_bpp , _, _, x_hat, _, _, y, z, string, side_string, _  = \
    build_model(x_scaled, lmbda, 'testing', msssim_loss=msssim_loss)  

  x_hat_to_save = x_hat
  x_hat = tf.cast(quantize_image(x_hat), tf.float32)

  mse = tf.squeeze(tf.reduce_mean(tf.squared_difference(x, x_hat), 
                                  axis=[1, 2, 3]))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))
  
  img_file_name = tf.placeholder(tf.string)
  save_reconstructed_op = write_png(img_file_name, x_hat_to_save[0])

  lossl, msel, psnrl, msssiml, msssim_dbl, eval_bppl, bppl = \
                                      [], [], [], [], [], [], []
    
  with tf.Session() as sess:
    tf.train.Saver().restore(sess, save_path=trained_model)
    for i in range(len(x_val_files)):
      test_file_name = \
        "/".join(x_val_files[i].split("/")[dataset_constants["path_prefix"]+1:])
      reconstucted_im_path = os.path.join(compressed_reconstructed_dir,
                              test_file_name[:-4] +'.png')
      im_metrics_path      = os.path.join(compressed_reconstructed_dir,
                              test_file_name[:-4] + '_metrics'+'.pkl')
      tensors = [string, side_string, tf.shape(x)[1:-1],
                 tf.shape(y)[1:-1], tf.shape(z)[1:-1]]
      if save_images:
        loss_, eval_bpp_, mse_, psnr_, msssim_, arrays, _ = \
          sess.run([loss, eval_bpp, mse, psnr, msssim, tensors,
                     save_reconstructed_op],
                      feed_dict={img_file_name:reconstucted_im_path})
      else:
        loss_, eval_bpp_, mse_, psnr_, msssim_, arrays = \
          sess.run([loss, eval_bpp, mse, psnr, msssim, tensors],
                    feed_dict={img_file_name:reconstucted_im_path})

      packed = PackedTensors()
      packed.pack(tensors, arrays)
      bpp_ = len(packed.string) * 8 / np.prod(dataset_constants["image_shape"])
    
      msssim_db_ = (-10 * np.log10(1 - msssim_))
      im_metrics = {'mse': mse_, 'psnr': psnr_, 'msssim': msssim_,
                    'msssim_db': msssim_db_, 'eval_bpp': eval_bpp_, 'bpp': bpp_}

      if save_images:    
        with open(im_metrics_path, "wb") as fp:
          pickle.dump(im_metrics, fp)
      
      lossl.append(loss_)
      msel.append(mse_)
      psnrl.append(psnr_)
      msssiml.append(msssim_)
      msssim_dbl.append(msssim_db_)
      eval_bppl.append(eval_bpp_)
      bppl.append(bpp_)
    
    loss_ = np.mean(lossl)
    mse_ = np.mean(msel)
    psnr_ = np.mean(psnrl)
    msssim_ = np.mean(msssiml)
    eval_bpp_ = np.mean(eval_bppl)
    bpp_ = np.mean(bppl)
    msssim_db_ = np.mean(msssim_dbl)

    exp_avg_metrics = {'loss': loss_, 'mse': mse_, 'psnr': psnr_, 
                       'msssim': msssim_, 'msssim_db': msssim_db_,
                       'eval_bpp': eval_bpp_, 'bpp': bpp_}

    print(exp_avg_metrics)

    if save_images:    
      with open(metrics_path, "wb") as fp:
        pickle.dump({'exp_avg_metrics': exp_avg_metrics}, fp)


def generate_dataset(dataset_constants,
               lmbda,
               trained_model,
               save_dir,
               split="train",
               evaluate=False,
               batchsize=8,
               msssim_loss=False):
  
  print("Generating dataset from {}".format(trained_model))

  if evaluate:
    evaluation(dataset_constants,
               lmbda,
               trained_model,
               save_dir,
               save_images=True,
               split=split,
               msssim_loss=msssim_loss)
    return

  def set_shape(x):
    x.set_shape(dataset_constants["image_shape"] + [3])
    return x
  
  val_preprocess_threads = 8
  test_glob = dataset_constants["{}_images_glob".format(split)]
  x_val_files = glob.glob(test_glob)
  val_dataset = tf.data.Dataset.from_tensor_slices(x_val_files)
  val_dataset = val_dataset.map(read_png, 
                                num_parallel_calls=val_preprocess_threads)
  val_dataset = val_dataset.map(set_shape,
                                num_parallel_calls=val_preprocess_threads)
  val_dataset = val_dataset.batch(batchsize)
  val_dataset = val_dataset.prefetch(batchsize)
  x = val_dataset.make_one_shot_iterator().get_next()    
  x_scaled = x / 255.

  _, _ , _, _, x_hat, _, _, _, _, _, _, _  = \
    build_model(x_scaled, lmbda, 'testing', msssim_loss=msssim_loss)  
  
  x_hat = quantize_image(x_hat)  

  with tf.Session() as sess:
    tf.train.Saver().restore(sess, save_path=trained_model)
    for i in range(math.ceil(len(x_val_files) / batchsize)):
      files = x_val_files[i*batchsize:(i+1)*batchsize]
      images = sess.run(x_hat)       
      for j in range(len(files)):      
        test_file_name = \
          "/".join(files[j].split("/")[dataset_constants["path_prefix"]+1:])
        reconstructed_im_path = os.path.join(save_dir,
                                test_file_name[:-4] +'.png')
        rec_img = images[j][:, :, ::-1]
        folder = "/".join(reconstructed_im_path.split("/")[:-1])
        if not os.path.exists(folder):
          os.makedirs(folder)
        cv2.imwrite(reconstructed_im_path, rec_img)
