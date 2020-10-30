# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
import argparse
import pickle
# from tqdm import tqdm
from training.misc import progress as tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

from perceptual_model import PerceptualModel
from utils.logger import setup_logger
from utils.visualizer import adjust_pixel_range
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('image_list', type=str,
                      help='List of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size. (default: 32)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--radius', type=int, default=3,
                      help='radius of each latent w')
  parser.add_argument('--start', type=int, default=0,
                      help='decide start from which images')
  parser.add_argument('--end', type=int, default=1,
                      help='decide end with which images')
  parser.add_argument('--attr', type=str, default='male',
                      help='')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  tflib.init_tf({'rnd.np_random_seed': 1000})
  with open(args.model_path, 'rb') as f:
    E, _, _, Gs = pickle.load(f)
  # Get input size.
  image_size = E.input_shape[2]
  assert image_size == E.input_shape[3]
  input_shape = E.input_shape
  num_layers, z_dim = Gs.components.synthesis.input_shape[1:3]

  ATTRS = {'age': [1,2,3], 'pose': [0,1,2], 'male': [2,3,4,5], 'expression': [3,4,5], 'glass': [1,2,3]}
  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
  latent_w = tf.placeholder(tf.float32, shape=[args.batch_size, num_layers, z_dim], name='latent_w')
  flag = tf.placeholder(tf.float32, shape=[], name='flag')
  w_enc = E.get_output_for(x, is_training=False)
  w_enc = tf.reshape(w_enc, [-1, 14, 512])
  coef = args.radius / np.sqrt(z_dim / 3.0)
  if args.attr == 'age':
    inds = ATTRS.get('age')
    aug = tf.zeros_like(latent_w)
    uniform = tf.random.uniform(shape=[args.batch_size, len(ATTRS.get('age')), z_dim], minval=-1.0, maxval=1.0) * coef
    aug_square = tf.concat([aug[:, :inds[0]], uniform, aug[:, inds[-1]+1:]], axis=1)
    latent_wp_square = latent_w + flag * aug_square
  elif args.attr == 'pose':
    inds = ATTRS.get('pose')
    aug = tf.zeros_like(latent_w)
    uniform = tf.random.uniform(shape=[args.batch_size, len(ATTRS.get('pose')), z_dim], minval=-1.0, maxval=1.0) * coef
    aug_square = tf.concat([aug[:, :inds[0]], uniform, aug[:, inds[-1]+1:]], axis=1)
    latent_wp_square = latent_w + flag * aug_square
  elif args.attr == 'male':
    inds = ATTRS.get('male')
    aug = tf.zeros_like(latent_w)
    uniform = tf.random.uniform(shape=[args.batch_size, len(ATTRS.get('male')), z_dim], minval=-1.0, maxval=1.0) * coef
    aug_square = tf.concat([aug[:, :inds[0]], uniform, aug[:, inds[-1]+1:]], axis=1)
    latent_wp_square = latent_w + flag * aug_square
  elif args.attr == 'expression':
    inds = ATTRS.get('expression')
    aug = tf.zeros_like(latent_w)
    uniform = tf.random.uniform(shape=[args.batch_size, len(ATTRS.get('expression')), z_dim], minval=-1.0, maxval=1.0) * coef
    aug_square = tf.concat([aug[:, :inds[0]], uniform, aug[:, inds[-1]+1:]], axis=1)
    latent_wp_square = latent_w + flag * aug_square
  elif args.attr == 'glass':
    inds = ATTRS.get('glass')
    aug = tf.zeros_like(latent_w)
    uniform = tf.random.uniform(shape=[args.batch_size, len(ATTRS.get('glass')), z_dim], minval=-1.0, maxval=1.0) * coef
    aug_square = tf.concat([aug[:, :inds[0]], uniform, aug[:, inds[-1]+1:]], axis=1)
    latent_wp_square = latent_w + flag * aug_square
  else:
    raise ValueError('no attr supported!!!')
  x_rec = Gs.components.synthesis.get_output_for(latent_wp_square, randomize_noise=False)
  # Load image list.
  logger.info(f'Loading image list.')
  image_list = []
  with open(args.image_list, 'r') as f:
    for line in f:
      image_list.append(line.strip())
  image_list = image_list[args.start:args.end]
  images = []
  names  = []
  for image_name in image_list:
    image = load_image(image_name)
    images.append(np.transpose(image, [2, 0, 1]))
    names.append(os.path.splitext(os.path.basename(image_name))[0])
  images = np.asarray(images)
  logger.info(f'images shape {images.shape}')
  images = images.astype(np.float32) / 255 * 2.0 - 1.0
  for idx in range(images.shape[0]):
    imgs = images[idx:idx+1]
    w_enc_ = sess.run(w_enc, {x: imgs})
    w_enc_ = np.tile(w_enc_, [args.batch_size, 1, 1])
    x_rec_ = sess.run(x_rec, {latent_w: w_enc_, flag: 0})
    imgs_ = adjust_pixel_range(x_rec_)
    for ii in range(2):
      save_image(f'{output_dir}/{names[idx]}_enc_{ii:04d}.png', imgs_[ii])
    for i in range(10):
      x_rec_ = sess.run(x_rec, {latent_w: w_enc_, flag: 1})
      imgs_ = adjust_pixel_range(x_rec_)
      for ii in range(imgs_.shape[0]):
        save_image(f'{output_dir}/{names[idx]}_rand_{i*10 +ii:04d}.png', imgs_[ii])




if __name__ == '__main__':
  main()
