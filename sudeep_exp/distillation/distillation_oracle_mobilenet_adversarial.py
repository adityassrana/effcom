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
from spectral_conv import ConvSN2D
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


def blockLayer(x, channels, r, kernel_size=[3,3]):
    output = tf.layers.conv2d(x, channels, (3, 3), padding='same', dilation_rate=(r, r), use_bias=False)
    return tf.nn.relu(output)

def resDenseBlock(x, channels=64, layers=8, kernel_size=[3,3], scale=1, growth_rate=16):
    outputs = [x]
    rates = [1,1,1,1,1,1,1,1]
    for i in range(layers):
        output = blockLayer(tf.concat(outputs[:i],3) if i>=1 else x, channels+i*growth_rate, rates[i])
        outputs.append(output)

    output = tf.concat(outputs, axis=-1)
    output = tf.layers.conv2d(output, channels, [1,1], padding="same", use_bias=False)
    output *= scale
    return x + output
  
class RDN(tf.keras.layers.Layer):
    def __init__(self, global_layers=6, local_layers=4, feature_size=64, growth_rate=8, *args, **kwargs):
        self.global_layers = global_layers
        self.local_layers = local_layers
        self.feature_size = feature_size
        self.growth_rate = growth_rate
        super(RDN, self).__init__(*args, **kwargs)

    def call(self, x):        
        scaling_factor = 0.1
        img = x

        x1 = tf.layers.conv2d(x,self.feature_size, [3,3], padding="same", use_bias=False)

        x = tf.layers.conv2d(x1, self.feature_size, [3,3], padding="same", use_bias=False)

        outputs = []
        for i in range(self.global_layers):
            x = resDenseBlock(x, self.feature_size, layers=self.local_layers, scale=scaling_factor,growth_rate=self.growth_rate)
            outputs.append(x)

        x = tf.concat(outputs, axis=-1)
        x = tf.layers.conv2d(x, self.feature_size, [1, 1], padding="same", use_bias=False)
        x = tf.layers.conv2d(x, self.feature_size, [3, 3], padding="same", use_bias=False)

        x = x + x1

        #x = utils.upsample(x,scale,feature_size)
        #output = slim.conv2d(x,output_channels,[3,3])
        res = tf.layers.conv2d(x, 3, (3, 3), padding='same', use_bias=False)
        return img + res


class SNResBlock(tf.keras.layers.Layer):
  def __init__(self, channels, kernel, downsample=True, *args, **kwargs):
    self.channels = channels
    self.kernel = kernel
    self.downsample = downsample
    super(SNResBlock, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
      tf.keras.layers.Activation("relu"),
      ConvSN2D(self.channels, self.kernel, strides=(1,1), padding="same", use_bias=False),
      tf.keras.layers.Activation("relu"),
      ConvSN2D(self.channels, self.kernel, strides=(1,1), padding="same", use_bias=False),
    ]
    super(SNResBlock, self).build(input_shape)

  def call(self, x):
    shortcut = x
    for layer in self._layers:
      x = layer(x)
    if self.downsample:
      return tf.keras.layers.AveragePooling2D()(x + shortcut)    
    return x + shortcut

class GlobalSumPooling2D(tf.keras.layers.Layer):
  def call(self, inputs):
    return tf.reduce_sum(inputs, axis=[1, 2])

class PatchDiscriminator(tf.keras.layers.Layer):
  def __init__(self, patchsize, *args, **kwargs):
    self.patchsize = patchsize
    assert patchsize in [32, 64, 128, 256]
    super(PatchDiscriminator, self).__init__(*args, **kwargs)
  
  def build(self, input_shape):
    self.h, self.w = input_shape[1], input_shape[2]
    patchsize = self.patchsize
    pad_h, pad_w = (patchsize - (self.h % patchsize)) % patchsize, (patchsize - (self.w % patchsize)) % patchsize
    self.paddings = [[0,0], [0, pad_h], [0, pad_w],  [0,0]]
    self.h += pad_h
    self.w += pad_w
    self._layers = []     
    if patchsize == 256:
      self._layers.append(ConvSN2D(64, 5, strides=(2, 2), padding="same", use_bias=False))
      self._layers.append(ConvSN2D(128, 3, strides=(2, 2), padding="same", use_bias=False))
      self._layers.append(SNResBlock(128, 3))
    elif patchsize == 128:
      self._layers.append(ConvSN2D(128, 3, strides=(2, 2), padding="same", use_bias=False))
      self._layers.append(SNResBlock(128, 3))
    elif patchsize == 64:
      self._layers.append(ConvSN2D(128, 3, padding="same", use_bias=False))
      self._layers.append(SNResBlock(128, 3))
    elif patchsize == 32:
      self._layers.append(ConvSN2D(128, 3, padding="same", use_bias=False))
    self._layers += [
      SNResBlock(128, 3),
      SNResBlock(128, 3),
      SNResBlock(128, 3),
      GlobalSumPooling2D(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dense(1)
    ]
    super(PatchDiscriminator, self).build(input_shape)
    
  def call(self, x, condition):
    inputs = tf.concat([x, condition], axis=-1)
    inputs = tf.pad(inputs, self.paddings)
    inputs = tf.concat([
        inputs[:, i: i + self.patchsize, j : j + self.patchsize, :] for i in range(0, self.h, self.patchsize) for j in range(0, self.w, self.patchsize)],
        axis=0)
    for layer in self._layers:
      inputs = layer(inputs)
    return inputs

def log_all_summaries(in_imgs, x_tilde, y, seg_logits, seg_labels, loss, train_bpp, train_mse, seg_loss, ssim, distillation, wasserstein_distance, l1, mode):
    summaries = []    
    summaries.append(tf.summary.image(mode+'_input_image', in_imgs)) # 
    #tf.summary.image("input_rgb", quantize_image(in_imgs))
    summaries.append(tf.summary.scalar(mode+'_loss', loss))
    if y is not None:
        summaries.append(tf.summary.image(mode+"_recon", quantize_image(y)))
        summaries.append(tf.summary.image(mode+'_diff', quantize_image(10*tf.math.abs(x_tilde - y))))
        summaries.append(tf.summary.scalar(mode+"_l1_diff", tf.reduce_mean(255.0 * tf.math.abs(x_tilde - y))))
    if train_bpp is not None:
        summaries.append(tf.summary.scalar(mode+"_bpp", train_bpp))
    if train_mse is not None:
        summaries.append(tf.summary.scalar(mode+"_mse", train_mse * (255 ** 2)))
    if seg_loss is not None:
        summaries.append(tf.summary.scalar(mode+"_seg_cross_entropy", seg_loss))
    if x_tilde is not None:
        summaries.append(tf.summary.image(mode+"_processed_reconstruction", quantize_image(x_tilde)))
    if ssim is not None:
        summaries.append(tf.summary.scalar(mode+"_ms-ssim (dB)", -10*tf.log(ssim) / np.log(10)))  
    if distillation is not None:  
        summaries.append(tf.summary.scalar(mode+"_distillation", distillation))
    if l1 is not None:  
        summaries.append(tf.summary.scalar(mode+"_l1", l1))
    if wasserstein_distance is not None:
       summaries.append(tf.summary.scalar(mode+"_wasserstein_distance", wasserstein_distance))
        
    if (seg_logits is not None) and (seg_labels is not None):
        cityscapes_label_colormap = get_dataset_colormap.create_cityscapes_label_colormap()
        cmp = tf.convert_to_tensor(cityscapes_label_colormap, tf.int32)  # (256, 3)
        predictions = tf.expand_dims(tf.argmax(seg_logits, 3), -1)
        summary_predictions = tf.gather(params=cmp, indices=predictions[:,:, :,0])
        summary_label = tf.gather(params=cmp, indices=seg_labels[:,:, :,0])
        semantic_map = tf.cast(summary_predictions, tf.uint8)
        seg_gt = tf.cast(summary_label, tf.uint8)

        summaries.append(tf.summary.image(mode+"_semantic_map", semantic_map))
        summaries.append(tf.summary.image(mode+"_label", seg_gt))
    return tf.summary.merge(summaries)

def train(l_args):
  """Trains the model."""
  if l_args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  x_train_files = sorted(glob.glob('/datatmp/Datasets/Cityscapes/leftImg8bit/train/*/*.png'))
  x_label_files = sorted(glob.glob('/datatmp/Datasets/Cityscapes/gtFine/train/*/*_labelIds.png'))
  y_train_files = sorted(glob.glob('/datatmp/Experiments/semantic_compression/{}/lambda_{}/leftImg8bit/train/*/*.png'.format(l_args.images_dir, l_args.lmbda)))
  
  print(len(x_train_files), len(y_train_files))  
  assert(len(x_train_files) == len(y_train_files))
  assert(x_train_files[0].split("/")[-1] == y_train_files[0].split("/")[-1])  
  assert(x_train_files[-1].split("/")[-1] == y_train_files[-1].split("/")[-1])    
  
  print(x_train_files[0].split("/")[-1][:-16], x_label_files[0].split("/")[-1].split("_gtFine_labelIds.png")[0])
  print(x_train_files[-1].split("/")[-1][:-16], x_label_files[-1].split("/")[-1].split("_gtFine_labelIds.png")[0])
  assert(len(x_label_files) == len(x_train_files))
  assert(x_train_files[0].split("/")[-1][:-16] == x_label_files[0].split("/")[-1].split("_gtFine_labelIds.png")[0])  
  assert(x_train_files[-1].split("/")[-1][:-16] == x_label_files[-1].split("/")[-1].split("_gtFine_labelIds.png")[0])    
  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train_files, x_label_files, y_train_files))
  train_dataset = train_dataset.shuffle(buffer_size=len(x_train_files)).repeat()
  train_dataset = train_dataset.map(read_pngs, num_parallel_calls=min(l_args.preprocess_threads, l_args.batchsize))
  if l_args.resize_images:
      train_dataset = train_dataset.map(lambda x : tf.image.resize_images(x, [512, 1024]))
  train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, [int(z) for z in l_args.patchsize.split(",")] + [7]))
  train_dataset = train_dataset.batch(l_args.batchsize)
  train_dataset = train_dataset.prefetch(l_args.batchsize)
  train_batch = train_dataset.make_one_shot_iterator().get_next()

  train_x, _, train_y = train_batch[:, :, :, :3] , train_batch[:, :, :, 3:4], train_batch[:, :, :, 4:]
  scaled_train_x, scaled_train_y = train_x / 255., train_y / 255.
  
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
  val_dataset = val_dataset.map(read_pngs, num_parallel_calls=1)
  val_dataset = val_dataset.map(set_shape, num_parallel_calls=1)
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

  x_features, _ = model.extract_features(train_x, model_options)
  exclude_list = ['global_step']
  variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude_list)
  seg_saver = tf.train.Saver(variables_to_restore)

  print(variables_to_restore)

  rdn = RDN()
  scaled_x_tilde_hat = rdn(scaled_train_y)
  x_tilde_hat = 255.0 * scaled_x_tilde_hat

  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    x_tilde_hat_features, _ = model.extract_features(x_tilde_hat, model_options)

  var_list = [var for var in tf.trainable_variables() if var not in variables_to_restore]
  
  print()
  print()
  print(var_list)

  discriminator = PatchDiscriminator(l_args.disc_patchsize)
  fake = tf.reduce_mean(discriminator(scaled_x_tilde_hat, scaled_train_y))
  real = tf.reduce_mean(discriminator(scaled_train_x, scaled_train_y))

  generator_loss = -1.0 * fake
  wasserstein_distance = real - fake
  discriminator_loss = -1.0 * wasserstein_distance

  mse = tf.reduce_mean(tf.squared_difference(scaled_train_x, scaled_x_tilde_hat)) * 255 ** 2
  ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(scaled_x_tilde_hat, scaled_train_x, 1))
  l1 = tf.reduce_mean(tf.math.abs(scaled_train_x - scaled_x_tilde_hat))

  distortion = {"mse" : mse, "l1":l1, "msssim":ssim, "msssim_l1":2*l1 + ssim}[l_args.loss_type]
  distillation = tf.reduce_mean(tf.squared_difference(x_features, x_tilde_hat_features))
  
  train_loss = l_args.rho * distortion + l_args.mu * distillation + generator_loss  
  
  rdn_weights = var_list
  discriminator_weights = discriminator.weights

  print()
  print()
  print(discriminator_weights)
  
  print()
  print()
  print([var for var in tf.trainable_variables() if var not in variables_to_restore + rdn_weights + discriminator_weights])
  
  step = tf.train.get_or_create_global_step()
  generator_optimizer = tf.train.AdamOptimizer(learning_rate=l_args.lr, beta1=0, beta2=0.9)
  generator_op = generator_optimizer.minimize(train_loss, var_list=rdn_weights, global_step=step)
  
  discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=l_args.disc_lr, beta1=0, beta2=0.9)
  discriminator_op = discriminator_optimizer.minimize(discriminator_loss, var_list=discriminator_weights)

  train_summary = log_all_summaries(train_x, scaled_x_tilde_hat,
               scaled_train_y, None, None, train_loss, None, mse, None, ssim, distillation, wasserstein_distance, l1, "train")


  scaled_x_val_hat = rdn(scaled_val_y)
  val_fake = tf.reduce_mean(discriminator(scaled_x_val_hat, scaled_val_y))
  val_real = tf.reduce_mean(discriminator(scaled_val_x, scaled_val_y))
  
  val_generator_loss = -1.0 * val_fake
  val_wasserstein = val_real - val_fake  
  
  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    val_x_features, _ = model.extract_features(val_x, model_options)
    x_val_hat_features, _ = model.extract_features(255.0*scaled_x_val_hat, model_options)
  
  val_distillation = tf.reduce_mean(tf.squared_difference(val_x_features, x_val_hat_features))
  val_mse = tf.reduce_mean(tf.squared_difference(scaled_val_x, scaled_x_val_hat)) * 255 ** 2
  val_ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(scaled_x_val_hat, scaled_val_x, 1))
  val_l1 = tf.reduce_mean(tf.math.abs(scaled_val_x - scaled_x_val_hat))
  val_distortion = {"mse" : val_mse, "l1":val_l1, "msssim":val_ssim, "msssim_l1":2*val_l1 + val_ssim}[l_args.loss_type]
  val_loss = l_args.rho * val_distortion + l_args.mu * val_distillation + val_generator_loss    

  #valid_summary = log_all_summaries(val_x, scaled_x_val_hat, scaled_val_y,
  #                 None, None, val_loss, None, val_mse, None, val_ssim, val_distillation, val_wasserstein, val_l1, "val")
  
  def load_pretrain(scaffold, sess):
    seg_saver.restore(sess, save_path=PATH_TO_TRAINED_MODEL)

  hooks = [
      tf.train.StopAtStepHook(last_step=l_args.last_step),
      tf.train.NanTensorHook(train_loss),
      #tf.train.SummarySaverHook(save_secs=120, output_dir=l_args.checkpoint_dir,summary_op=valid_summary),
      tf.train.SummarySaverHook(save_secs=60,output_dir=l_args.checkpoint_dir,summary_op=train_summary),
  ]
  
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=l_args.checkpoint_dir,
      save_checkpoint_secs=1200, save_summaries_steps=None, save_summaries_secs=None, scaffold=tf.train.Scaffold(init_fn=load_pretrain)) as sess:    
    while not sess.should_stop():      
      sess.run(discriminator_op)
      sess.run(generator_op)

def evaluation(l_args):
    lmbda = l_args.lmbda
    train_dir = l_args.checkpoint_dir
    metrics_path = os.path.join(train_dir, 'metrics_args.pkl')
    l_args.lmbda = lmbda
    compressed_reconstructed_dir = os.path.join(train_dir, 'compressed_reconstructed_images')
    if not os.path.exists(compressed_reconstructed_dir):
        os.makedirs(compressed_reconstructed_dir)
    val_split_size = 500

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
    
    layers = RDN()
    scaled_x_hat = layers(scaled_val_y)
    x_hat = 255.0 * scaled_x_hat
    #x_hat = tf.clip_by_value(scaled_x_tilde_hat, 0, 1)
    x_hat = tf.clip_by_value(tf.round(x_hat), 0, 255)

    mse = tf.reduce_mean(tf.squared_difference(val_x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x_hat, val_x, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, val_x, 255))
    # Write reconstructed image out as a PNG file.
    img_file_name = tf.placeholder(tf.string)
    save_reconstructed_op = write_png(img_file_name, scaled_x_hat[0])

    logger.info('Testing the model on ' + str(val_split_size) + ' images and save the reconstructed images')
    msel, psnrl, msssiml, msssim_dbl, eval_bppl, bppl = [], [], [], [], [], []

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        try:
          latest = tf.train.latest_checkpoint(checkpoint_dir=l_args.checkpoint_dir)
        except:
          latest = None
        if not latest:
          latest = os.path.join(l_args.checkpoint_dir, "model.ckpt-{}".format(l_args.last_step))        
        best_chekpnt = latest
        tf.train.Saver().restore(sess, save_path=best_chekpnt)

        for i in range(val_split_size):
            test_file_name       = "/".join(x_val_files[i].split("/")[4:])
            reconstucted_im_path = os.path.join(compressed_reconstructed_dir,test_file_name[:-4] + '_reconstructed'+'.png')
            im_metrics_path      = os.path.join(compressed_reconstructed_dir,test_file_name[:-4] + '_metrics'+'.pkl')
            l_args.output        = reconstucted_im_path

            mse_, psnr_, msssim_, _ = \
                sess.run([mse, psnr, msssim, save_reconstructed_op], feed_dict={img_file_name:reconstucted_im_path})
            #mse_, psnr_, msssim_, = sess.run([mse, psnr, msssim])

            msssim_db_ = (-10 * np.log10(1 - msssim_))

            im_metrics = {'mse': mse_, 'psnr': psnr_, 'msssim': msssim_, 'msssim_db': msssim_db_}
            with open(im_metrics_path, "wb") as fp:
                pickle.dump(im_metrics, fp)

            msel.append(mse_)
            psnrl.append(psnr_)
            msssiml.append(msssim_)
            msssim_dbl.append(msssim_db_)
            
    logger.info('Averaging metrics and save them with the exp_args in pickle file metrics_args.pkl' )
    mse_ = np.mean(msel)
    psnr_ = np.mean(psnrl)
    msssim_ = np.mean(msssiml)
    msssim_db_ = np.mean(msssim_dbl)

    logger.info('MSE        = ' + str(mse_))
    logger.info('PSNR       = ' + str(psnr_))
    logger.info('MS-SSIM    = ' + str(msssim_))
    logger.info('MS-SSIM db = ' + str(msssim_db_))
    exp_avg_metrics = {'mse': mse_, 'psnr': psnr_, 'msssim': msssim_, 'msssim_db': msssim_db_, 'chk_pnt':best_chekpnt}

    with open(metrics_path, "wb") as fp:
        pickle.dump({'exp_avg_metrics': exp_avg_metrics, 'exp_args': l_args}, fp)

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
      "--disc_patchsize", type=int, default=256,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lr", type=float, default=0.0001)
  parser.add_argument(
      "--disc_lr", type=float, default=0.0004)
  parser.add_argument(
      "--mu", type=float, default=0.1)
  parser.add_argument(
      "--rho", type=float, default=1.0)
  parser.add_argument(
      "--lmbda", type=float, default=0.002,
      help="list of lambda values that the model will be trained with")
  parser.add_argument(
      "--exp-name", default="rda_v3",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--exp-base", default="/datatmp/Experiments/semantic_compression",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--images-dir", default="recon_new_msh_ft_cityscapes",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--last_step", type=int, default=40000,
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
  parser.add_argument("--loss_type", type=str, default="msssim_l1")
  parser.add_argument("--resize_images", dest='resize_images', action='store_true')      
  parser.add_argument('--test_only', dest='test_only', action='store_true')
  
  # Parse arguments.  
  args = parser.parse_args(argv[1:])
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  return args

def experiment(args):  
  exp_dir = os.path.join(args.exp_base, args.exp_name)
  exp_dir = os.path.join(exp_dir, args.loss_type)
  args.checkpoint_dir = os.path.join(os.path.join(os.path.join(exp_dir, 'mu_{}'.format(args.mu)), 'rho_{}'.format(args.rho)), 'lambda_'+str(args.lmbda))
  
  if not args.test_only:
    train(args)
    tf.reset_default_graph()

  if not os.path.exists(args.checkpoint_dir):
    return

  evaluation(args)

def main(unused_argv):
  FLAGS.aspp_with_batch_norm = False
  FLAGS.aspp_with_separable_conv = False
  FLAGS.decoder_use_separable_conv = False
  app.run(experiment, flags_parser=parse_args)

if __name__ == "__main__":
  tf.app.run()
