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

def log_all_summaries(in_imgs, x_tilde, y, seg_logits, seg_labels, loss, train_bpp, train_mse, seg_loss, ssim, mode):
    tf.summary.image(mode+'_input_image', in_imgs) # 
    #tf.summary.image("input_rgb", quantize_image(in_imgs))
    tf.summary.scalar(mode+'_loss', loss)
    if y is not None:
        tf.summary.image(mode+"_recon", quantize_image(y))
        tf.summary.image(mode+'_diff', quantize_image(10*tf.math.abs(x_tilde - y)))
    if train_bpp is not None:
        tf.summary.scalar(mode+"_bpp", train_bpp)
    if train_mse is not None:
        tf.summary.scalar(mode+"_mse", train_mse * (255 ** 2))
    if seg_loss is not None:
        tf.summary.scalar(mode+"_seg_cross_entropy", seg_loss)
    if x_tilde is not None:
        tf.summary.image(mode+"_processed_reconstruction", quantize_image(x_tilde))
    if ssim is not None:
        tf.summary.scalar(mode+"_ms-ssim (dB)", -10*tf.log(ssim) / np.log(10))    

    if (seg_logits is not None) and (seg_labels is not None):
        cityscapes_label_colormap = get_dataset_colormap.create_cityscapes_label_colormap()
        cmp = tf.convert_to_tensor(cityscapes_label_colormap, tf.int32)  # (256, 3)
        predictions = tf.expand_dims(tf.argmax(seg_logits, 3), -1)
        summary_predictions = tf.gather(params=cmp, indices=predictions[:,:, :,0])
        summary_label = tf.gather(params=cmp, indices=seg_labels[:,:, :,0])
        semantic_map = tf.cast(summary_predictions, tf.uint8)
        seg_gt = tf.cast(summary_label, tf.uint8)

        tf.summary.image(mode+"_semantic_map", semantic_map)
        tf.summary.image(mode+"_label", seg_gt)


def build_model(x, y, loss_type):
  scaled_train_x = x
  scaled_train_y = y
  layers = RDN()
  scaled_x_tilde_hat = layers(scaled_train_y)
  mse = tf.reduce_mean(tf.squared_difference(scaled_train_x, scaled_x_tilde_hat)) * 255 ** 2
  ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(scaled_x_tilde_hat, scaled_train_x, 1))
  l1 = tf.reduce_mean(tf.math.abs(scaled_train_x - scaled_x_tilde_hat))
  distortion = {"mse" : mse, "l1":l1, "msssim":ssim}[loss_type]
  return scaled_x_tilde_hat, distortion

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
  train_dataset = train_dataset.map(read_pngs, num_parallel_calls=l_args.preprocess_threads)
  train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, [int(z) for z in l_args.patchsize.split(",")] + [7]))
  train_dataset = train_dataset.batch(l_args.batchsize)
  train_dataset = train_dataset.prefetch(l_args.batchsize)
  train_batch = train_dataset.make_one_shot_iterator().get_next()

  train_x, _, train_y = train_batch[:, :, :, :3] , train_batch[:, :, :, 3:4], train_batch[:, :, :, 4:]
  scaled_train_x, scaled_train_y = train_x / 255., train_y / 255.
  
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

  layers = RDN()
  scaled_x_tilde_hat = layers(scaled_train_y)
  x_tilde_hat = 255.0 * scaled_x_tilde_hat

  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    x_tilde_hat_features, _ = model.extract_features(x_tilde_hat, model_options)

  mse = tf.reduce_mean(tf.squared_difference(scaled_train_x, scaled_x_tilde_hat)) * 255 ** 2
  ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(scaled_x_tilde_hat, scaled_train_x, 1))
  l1 = tf.reduce_mean(tf.math.abs(scaled_train_x - scaled_x_tilde_hat))

  distortion = {"mse" : mse, "l1":l1, "msssim":ssim}[l_args.loss_type]
  distillation = tf.reduce_mean(tf.squared_difference(x_features, x_tilde_hat_features))
  
  train_loss = distortion + l_args.mu * distillation  
 
  var_list = [var for var in tf.trainable_variables() if var not in variables_to_restore]
  
  print()
  print()
  print(var_list)


  step = tf.train.get_or_create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=l_args.lr)
  train_op = main_optimizer.minimize(train_loss, var_list=var_list, global_step=step)

  log_all_summaries(train_x, scaled_x_tilde_hat, scaled_train_y, None, None, train_loss, None, mse, None, ssim, "train")
  #log_all_summaries(val_x, valid_x_tilde, None, None, valid_loss, None, valid_mse, None, valid_ssim, "val")

  hooks = [
      tf.train.StopAtStepHook(last_step=l_args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  
  def load_pretrain(scaffold, sess):
    seg_saver.restore(sess, save_path=PATH_TO_TRAINED_MODEL)

  hooks = [
      tf.train.StopAtStepHook(last_step=l_args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=l_args.checkpoint_dir,
      save_checkpoint_secs=1200, save_summaries_secs=60, scaffold=tf.train.Scaffold(init_fn=load_pretrain)) as sess:    
    while not sess.should_stop():      
      sess.run(train_op)

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
          latest = os.path.join(l_args.checkpoint_dir, "model.ckpt-{}".format(l_args.last_step))  
        best_checkpoint = latest
        tf.train.Saver().restore(sess, save_path=best_checkpoint)

        for i in range(val_split_size):
            test_file_name       = "/".join(x_val_files[i].split("/")[4:])
            reconstucted_im_path = os.path.join(compressed_reconstructed_dir,test_file_name+'_reconstructed'+'.png')
            im_metrics_path      = os.path.join(compressed_reconstructed_dir,test_file_name +'_metrics'+'.pkl')
            l_args.output        = reconstucted_im_path

            #tensors = [string, side_string, tf.shape(x)[1:-1], tf.shape(y)[1:-1], tf.shape(z)[1:-1]]
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
    exp_avg_metrics = {'mse': mse_, 'psnr': psnr_, 'msssim': msssim_, 'msssim_db': msssim_db_, 'chk_pnt':best_checkpoint}

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
      "--valid_patclinehsize", type=int, default=512)
  parser.add_argument(
      "--gpu", type=str, default='0',
      help="Gpu to use")
  parser.add_argument("--loss_type", type=str, default="msssim")   
  parser.add_argument('--test_only', dest='test_only', action='store_true')
  
  # Parse arguments.  
  args = parser.parse_args(argv[1:])
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  return args

def experiment(args):  
  exp_dir = os.path.join(args.exp_base, args.exp_name)
  exp_dir = os.path.join(exp_dir, args.loss_type)
  args.checkpoint_dir = os.path.join(os.path.join(exp_dir, 'mu_{}'.format(args.mu)), 'lambda_'+str(args.lmbda))
  
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
