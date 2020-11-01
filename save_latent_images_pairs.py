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
from training.misc import progress as tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

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
  parser.add_argument('--batch_size', type=int, default=50,
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
  parser.add_argument('--max_num_layers', type=int, default=8,
                      help='the maximum layer to replace')
  parser.add_argument('--total_nums', type=int, default=1e+5,
                      help='the maximum layer to replace')
  parser.add_argument('--save_raw', action='store_true',
                      help='Whether to save raw images')
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

  # ATTRS = {'age': [1,2,3], 'pose': [0,1,2], 'male': [2,3,4,5], 'expression': [3,4,5], 'glass': [1,2,3]}
  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
  latent_w = tf.placeholder(tf.float32, shape=[None, num_layers, z_dim], name='latent_w')
  w_enc = E.get_output_for(x, is_training=False)
  w_enc = tf.reshape(w_enc, [-1, 14, 512])
  coef = args.radius / np.sqrt(z_dim / 3.0)

  x_rec = Gs.components.synthesis.get_output_for(latent_w, randomize_noise=False)
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
  aug = np.zeros([args.batch_size, num_layers, z_dim])
  for idx in range(images.shape[0]):
    imgs = images[idx:idx+1]
    w_enc_ = sess.run(w_enc, {x: imgs})
    x_rec_ = sess.run(x_rec, {latent_w: w_enc_})
    imgs_ = adjust_pixel_range(x_rec_)
    if args.save_raw:
      save_image(f'{output_dir}/{names[idx]}_enc_init.png', imgs_[0])
    w_enc_ = np.tile(w_enc_, [args.batch_size, 1, 1])
    images_npy = []
    latent_w_s = []
    for it in tqdm(range(0, args.total_nums, args.batch_size)):
      num_layer_replace = np.random.randint(low=1, high=args.max_num_layers)
      start_layer_replace = np.random.randint(low=0, high=args.max_num_layers - num_layer_replace)
      end_layer_replace = start_layer_replace + num_layer_replace
      uniform = np.random.uniform(low=-1.0, high=1.0, size=[args.batch_size, num_layer_replace, z_dim]) * coef
      aug_square = np.concatenate([aug[:, :start_layer_replace], uniform, aug[:, end_layer_replace:]], axis=1)
      latent_w_square = w_enc_ + aug_square
      x_rec_ = sess.run(x_rec, {latent_w: latent_w_square})
      imgs_ = adjust_pixel_range(x_rec_)
      if args.save_raw:
        for ii in range(imgs_.shape[0]):
          save_image(f'{output_dir}/{names[idx]}_rand_{it +ii:04d}.png', imgs_[ii])
      latent_w_s.append(latent_w_square)
      images_npy.append(imgs_)
    latent_w_s = np.concatenate(latent_w_s, axis=0)
    images_npy = np.concatenate(images_npy, axis=0)
    logger.info(f'latent_w_s shape: {latent_w_s.shape}, Saving...')
    np.save(f'{output_dir}/{names[idx]}_latent.npy', latent_w_s)
    logger.info(f'images shape: {images_npy.shape}, Saving...')
    np.save(f'{output_dir}/{names[idx]}_image.npy', images_npy)


if __name__ == '__main__':
  main()