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

from perceptual_model import PerceptualModel
from training import misc
from utils.logger import setup_logger
from utils.visualizer import adjust_pixel_range
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path1', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('model_path2', type=str,
                      help='Path to the classifier model.')
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
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--loss_weight_feat', type=float, default=1e-3,
                      help='The perceptual loss scale for optimization. '
                           '(default: 1e-5)')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--reverse', action='store_true',
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
  assert os.path.exists(args.model_path1)
  assert os.path.exists(args.model_path2)
  E, _, _, Gs = misc.load_pkl(args.model_path1)
  classifier = misc.load_pkl(args.model_path2)

  # Get input size.
  image_size = E.input_shape[2]
  assert image_size == E.input_shape[3]
  input_shape = E.input_shape
  perceptual_model = PerceptualModel([image_size, image_size], False)

  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
  # latent_w = tf.placeholder(tf.float32, shape=[None, num_layers, z_dim], name='latent_w')
  w_enc = E.get_output_for(x, is_training=False)
  w_enc = tf.reshape(w_enc, [-1, 14, 512])
  wp = tf.get_variable(shape=[1, 14, 512], name='latent_code')
  x_rec = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
  setter = tf.assign(wp, w_enc)
  x_255 = (tf.transpose(x, [0, 2, 3, 1]) + 1) / 2 * 255
  x_rec_255 = (tf.transpose(x_rec, [0, 2, 3, 1]) + 1) / 2 * 255
  x_feat = perceptual_model(x_255)
  x_rec_feat = perceptual_model(x_rec_255)
  loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=[1])
  loss_feat = args.loss_weight_feat * loss_feat
  if args.reverse:
    scores = -classifier.get_output_for(x_rec, None)
  else:
    scores = classifier.get_output_for(x_rec, None)
  loss = loss_feat + scores
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_op = optimizer.minimize(loss, var_list=[wp])
  tflib.init_uninitialized_vars()

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
    sess.run(setter, {x: imgs})
    x_rec_ = sess.run(x_rec, {x: imgs})
    imgs_ = adjust_pixel_range(x_rec_)
    if args.save_raw:
      save_image(f'{output_dir}/{names[idx]}_enc_init.png', imgs_[0])
    for it in tqdm(range(0, args.num_iterations)):
      _, x_rec_, loss_, feat_loss_, scores_ = sess.run([train_op, x_rec, loss, loss_feat, scores], {x:imgs})
      print(f'Iter: {it:04d} loss: {loss_} feat_loss: {feat_loss_} score: {scores_}')
      imgs_ = adjust_pixel_range(x_rec_)
      if args.save_raw:
        for ii in range(imgs_.shape[0]):
          save_image(f'{output_dir}/{names[idx]}_edit_{it +ii:04d}.png', imgs_[ii])


if __name__ == '__main__':
  main()