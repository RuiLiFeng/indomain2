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
  parser.add_argument('image_dir', type=str,
                      help='Image directory, which includes original images, '
                           'inverted codes, and image list.')
  parser.add_argument('--cond_path', type=str, default='',
                      help='Path to the consitional classifier model.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size. (default: 4)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--start', type=int, default=0,
                      help='decide start from which images')
  parser.add_argument('--end', type=int, default=1,
                      help='decide end with which images')
  parser.add_argument('--save_raw', action='store_true',
                      help='Whether to save raw images')
  parser.add_argument('--num_iterations', type=int, default=50,
                      help='Number of optimization iterations. (default: 50)')
  parser.add_argument('--loss_weight_feat', type=float, default=1e-3,
                      help='The perceptual loss scale for optimization. '
                           '(default: 1e-3)')
  parser.add_argument('--loss_weight_pixel', type=float, default=20,
                      help='The pixel loss scale for optimization. '
                           '(default: 20)')
  parser.add_argument('--d_scale', type=float, default=1,
                      help='The discriminator loss scale for optimization. '
                           '(default: 1)')
  parser.add_argument('--min_values', type=float, default=-5,
                      help='The min score values for optimization')
  parser.add_argument('--min_values_cond', type=float, default=-4,
                      help='The min score values for optimization')
  parser.add_argument('--max_values', type=float, default=100,
                      help='The max score values for optimization')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--reverse', action='store_true',
                      help='Decide which direction to optimize')
  parser.add_argument('--reverse_cond', action='store_true',
                      help='Decide which direction to optimize')
  parser.add_argument('--num_images', type=int, default=10)
  parser.add_argument('--model_name', type=str, default='ffhq')
  parser.add_argument('--attr_name', type=str, default='')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  image_dir = args.image_dir
  image_dir_name = os.path.basename(image_dir.rstrip('/'))
  assert os.path.exists(image_dir)
  output_dir = args.output_dir or f'results/manipulation/'
  boundary_name = os.path.basename(args.model_path2).split('.')[0].split('-')[-1]
  c_boundary_name = 'foo'
  if os.path.exists(args.cond_path):
      c_boundary_name = os.path.basename(args.cond_path).split('.')[0].split('-')[-1]
  job_name = f'{boundary_name}_{image_dir_name}_reverse_{args.reverse}_cond_{c_boundary_name}_rev_{args.reverse_cond}'
  logger = setup_logger(output_dir, f'{job_name}.log', f'{job_name}_logger')
  ATTRS = {'age': [0, 1, 2, 3, 4],
           'pose': [0, 1, 2],
           'male': [1, 2, 3, 4, 5, 6],
           'expression': [2, 3, 4, 5],
           'addglass': [0, 1, 2, 3, 4, 5],
           'removeglass': [0, 1, 2],
           }
  logger.info(f'Loading model.')
  tflib.init_tf({'rnd.np_random_seed': 1000})
  assert os.path.exists(args.model_path1)
  assert os.path.exists(args.model_path2)
  E, _, D, Gs = misc.load_pkl(args.model_path1)
  classifier = misc.load_pkl(args.model_path2)
  if os.path.exists(args.cond_path):
    cond_classifier = misc.load_pkl(args.cond_path)

  # Get input size.
  image_size = E.input_shape[2]
  assert image_size == E.input_shape[3]
  input_shape = E.input_shape
  perceptual_model = PerceptualModel([image_size, image_size], False)
  num_layers, z_dim = Gs.components.synthesis.input_shape[1:3]
  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
  latent_w = tf.placeholder(tf.float32, shape=[None, num_layers, z_dim], name='latent_w')
  attr_related_layers = ATTRS.get(args.attr_name, list(range(8)))
  wps = []
  for i in range(num_layers):
    if i in attr_related_layers:
      trainable = True
    else:
      trainable = False
    latent_code = tf.get_variable(shape=[1, 1, z_dim],
                                  name=f'latent_code{i}', trainable=trainable)
    wps.append(latent_code)
  wp = tf.concat(wps, axis=1)
  x_rec = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
  setter_ops = []
  for i in range(num_layers):
    setter_ops.append(tf.assign(wps[i], latent_w[:, i:i+1]))
  setter = tf.group(setter_ops)
  code_to_optim = [v for v in tf.trainable_variables() if v.name.startswith("latent_code")]
  x_255 = (tf.transpose(x, [0, 2, 3, 1]) + 1) / 2 * 255
  x_rec_255 = (tf.transpose(x_rec, [0, 2, 3, 1]) + 1) / 2 * 255
  x_feat = perceptual_model(x_255)
  x_rec_feat = perceptual_model(x_rec_255)
  loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=1)
  loss_feat = args.loss_weight_feat * loss_feat
  loss_pixel = tf.reduce_mean(tf.square(x_rec - x), axis=[1, 2, 3])
  loss_pixel = args.loss_weight_pixel * loss_pixel
  if args.reverse:
    scores = -classifier.get_output_for(x_rec, None)
  else:
    scores = classifier.get_output_for(x_rec, None)
  scores = tf.clip_by_value(scores,
                            clip_value_min=args.min_values,
                            clip_value_max=args.max_values)
  cond_scores = tf.zeros_like(scores)
  if os.path.exists(args.cond_path):
    if args.reverse_cond:
      cond_scores = -cond_classifier.get_output_for(x_rec, None)
    else:
      cond_scores = cond_classifier.get_output_for(x_rec, None)
    cond_scores = tf.clip_by_value(cond_scores,
                                   clip_value_min=args.min_values_cond,
                                   clip_value_max=args.max_values)
  cond_scores = tf.reduce_mean(cond_scores, axis=1)
  scores = tf.reduce_mean(scores, axis=1)
  adv_score = D.get_output_for(x_rec, None)
  loss_adv = tf.reduce_mean(tf.nn.softplus(-adv_score), axis=1)
  loss_adv = args.d_scale * loss_adv
  loss = loss_feat + loss_pixel + scores + loss_adv + cond_scores
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_op = optimizer.minimize(loss, var_list=code_to_optim)
  tflib.init_uninitialized_vars()

  # Load image, codes, and boundary.
  logger.info(f'Loading images and corresponding inverted latent codes.')
  images_name = []
  images_orin = []
  images_invi = []
  with open(f'{image_dir}/image_list.txt', 'r') as f:
    for line in f:
      name = os.path.splitext(os.path.basename(line.strip()))[0]
      assert os.path.exists(f'{image_dir}/{name}_ori.png')
      assert os.path.exists(f'{image_dir}/{name}_inv.png')
      images_name.append(name)
      image = load_image(f'{image_dir}/{name}_ori.png')
      images_orin.append(np.transpose(image, [2, 0, 1]))
      image = load_image(f'{image_dir}/{name}_inv.png')
      images_invi.append(image)
  images_orin = np.asarray(images_orin)
  images_invi = np.asarray(images_invi)
  latent_codes = np.load(f'{image_dir}/inverted_codes.npy')
  assert latent_codes.shape[0] == images_orin.shape[0] == images_invi.shape[0]
  images_orin = images_orin.astype(np.float32) / 255 * 2.0 - 1.0
  images_orin = images_orin[args.start: args.end]
  images_invi = images_invi[args.start: args.end]
  latent_codes = latent_codes[args.start: args.end]

  save_interval = args.num_iterations // args.num_images
  headers = ['Name', 'Original Image', 'Inversion Image']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:04d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
    num_rows=images_orin.shape[0], num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  for idx in range(images_orin.shape[0]):
    imgs = images_orin[idx:idx+1]
    latent_code = latent_codes[idx:idx+1]
    sess.run(setter, {latent_w: latent_code})
    imgs_orin = adjust_pixel_range(imgs)
    visualizer.set_cell(idx, 0, text=images_name[idx])
    visualizer.set_cell(idx, 1, image=imgs_orin[0])
    visualizer.set_cell(idx, 2, image=images_invi[idx])
    col_idx = 3
    for it in range(1, args.num_iterations + 1):
      output_node = [train_op, loss, loss_feat, scores, cond_scores, loss_pixel, loss_adv]
      _, loss_, feat_loss_, scores_, cond_loss_, loss_pixel_, loss_adv_ = sess.run(output_node, {x: imgs})
      if it % save_interval == 0:
        x_rec_ = sess.run(x_rec)
        imgs_ = adjust_pixel_range(x_rec_)
        visualizer.set_cell(idx, col_idx, image=imgs_[0])
        col_idx += 1
        print(f'Iter: {it:04d} loss: {np.mean(loss_):6.4f} feat_loss: {np.mean(feat_loss_):6.4f}'
              f' pixel_loss: {np.mean(loss_pixel_):6.4f} score: {np.mean(scores_):6.4f}'
              f' cond_loss: {np.mean(cond_loss_):6.4f} adv: {np.mean(loss_adv_):6.4f}')
  visualizer.save(f'{output_dir}/{job_name}.html')


if __name__ == '__main__':
  main()
