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
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('src_dir', type=str,
                      help='Path to the classifier model.')
  parser.add_argument('dst_dir', type=str,
                      help='Image directory, which includes original images, '
                           'inverted codes, and image list.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size. (default: 1)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--src_start', type=int, default=0,
                      help='decide start from which images')
  parser.add_argument('--src_end', type=int, default=1,
                      help='decide end with which images')
  parser.add_argument('--dst_start', type=int, default=0,
                      help='decide start from which images')
  parser.add_argument('--dst_end', type=int, default=1,
                      help='decide end with which images')
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
  parser.add_argument('--latent_scale', type=float, default=1,
                      help='The latent loss scale for optimization. '
                           '(default: 1)')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_images', type=int, default=10)
  parser.add_argument('--model_name', type=str, default='ffhq')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  src_dir = args.src_dir
  src_dir_name = os.path.basename(src_dir.rstrip('/'))
  assert os.path.exists(src_dir)
  assert os.path.exists(f'{src_dir}/image_list.txt')
  assert os.path.exists(f'{src_dir}/inverted_codes.npy')
  dst_dir = args.dst_dir
  dst_dir_name = os.path.basename(dst_dir.rstrip('/'))
  assert os.path.exists(dst_dir)
  assert os.path.exists(f'{dst_dir}/image_list.txt')
  assert os.path.exists(f'{dst_dir}/inverted_codes.npy')
  output_dir = args.output_dir or 'results/interpolation'
  job_name = f'{src_dir_name}_TO_{dst_dir_name}'
  logger = setup_logger(output_dir, f'{job_name}.log', f'{job_name}_logger')

  logger.info(f'Loading model.')
  tflib.init_tf({'rnd.np_random_seed': 1000})
  assert os.path.exists(args.model_path)
  E, _, D, Gs = misc.load_pkl(args.model_path)

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
  latent_src = tf.placeholder(tf.float32, shape=[args.batch_size, num_layers, z_dim], name='latent_src')
  latent_dst = tf.placeholder(tf.float32, shape=[args.batch_size, num_layers, z_dim], name='latent_dst')
  wp = tf.get_variable(shape=[args.batch_size, num_layers, z_dim], name='latent_code')
  x_rec = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
  setter = tf.assign(wp, latent_src)
  x_255 = (tf.transpose(x, [0, 2, 3, 1]) + 1) / 2 * 255
  x_rec_255 = (tf.transpose(x_rec, [0, 2, 3, 1]) + 1) / 2 * 255
  x_feat = perceptual_model(x_255)
  x_rec_feat = perceptual_model(x_rec_255)
  loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=1)
  loss_feat = args.loss_weight_feat * loss_feat
  loss_pixel = tf.reduce_mean(tf.square(x_rec - x), axis=[1, 2, 3])
  loss_pixel = args.loss_weight_pixel * loss_pixel
  adv_score = D.get_output_for(x_rec, None)
  loss_adv = tf.reduce_mean(tf.nn.softplus(-adv_score), axis=1)
  loss_adv = args.d_scale * loss_adv
  w_loss = args.latent_scale * tf.reduce_mean(tf.square(wp - latent_dst))
  loss = loss_feat + loss_pixel + loss_adv + w_loss
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_op = optimizer.minimize(loss, var_list=[wp])
  tflib.init_uninitialized_vars()

  # Load image, codes, and boundary.
  logger.info(f'Loading images and corresponding inverted latent codes.')
  src_images_name = []
  src_images_orin = []
  src_images_invi = []
  with open(f'{src_dir}/image_list.txt', 'r') as f:
    for line in f:
      name = os.path.splitext(os.path.basename(line.strip()))[0]
      assert os.path.exists(f'{src_dir}/{name}_ori.png')
      assert os.path.exists(f'{src_dir}/{name}_inv.png')
      src_images_name.append(name)
      image = load_image(f'{src_dir}/{name}_ori.png')
      src_images_orin.append(np.transpose(image, [2, 0, 1]))
      image = load_image(f'{src_dir}/{name}_inv.png')
      src_images_invi.append(image)
  src_images_orin = np.asarray(src_images_orin)
  src_images_invi = np.asarray(src_images_invi)
  src_latent_codes = np.load(f'{src_dir}/inverted_codes.npy')
  assert src_latent_codes.shape[0] == src_images_orin.shape[0] == src_images_invi.shape[0]
  src_images_orin = src_images_orin.astype(np.float32) / 255 * 2.0 - 1.0
  src_images_orin = src_images_orin[args.src_start: args.src_end]
  src_images_invi = src_images_invi[args.src_start: args.src_end]
  src_latent_codes = src_latent_codes[args.src_start: args.src_end]
  num_src = args.src_end - args.src_start

  dst_images_name = []
  dst_images_orin = []
  dst_images_invi = []
  with open(f'{dst_dir}/image_list.txt', 'r') as f:
    for line in f:
      name = os.path.splitext(os.path.basename(line.strip()))[0]
      assert os.path.exists(f'{dst_dir}/{name}_ori.png')
      assert os.path.exists(f'{dst_dir}/{name}_inv.png')
      dst_images_name.append(name)
      image = load_image(f'{dst_dir}/{name}_ori.png')
      dst_images_orin.append(np.transpose(image, [2, 0, 1]))
      image = load_image(f'{dst_dir}/{name}_inv.png')
      dst_images_invi.append(image)
  dst_images_orin = np.asarray(dst_images_orin)
  dst_images_invi = np.asarray(dst_images_invi)
  dst_latent_codes = np.load(f'{dst_dir}/inverted_codes.npy')
  assert dst_latent_codes.shape[0] == dst_images_orin.shape[0] == dst_images_invi.shape[0]
  dst_images_orin = dst_images_orin.astype(np.float32) / 255 * 2.0 - 1.0
  dst_images_orin = dst_images_orin[args.dst_start: args.dst_end]
  dst_images_invi = dst_images_invi[args.dst_start: args.dst_end]
  dst_latent_codes = dst_latent_codes[args.dst_start: args.dst_end]
  num_dst = args.dst_end - args.dst_start

  save_interval = args.num_iterations // args.num_images
  headers = ['Name', 'Original Image', 'Inversion Image']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:04d}')
  headers.append('Target Image')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
    num_rows=src_images_orin.shape[0] * dst_images_orin.shape[0],
    num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  for src_ind in range(num_src):
    img_src = src_images_orin[src_ind:src_ind+1]
    img_src = adjust_pixel_range(img_src)
    latent_code_src = src_latent_codes[src_ind:src_ind + 1]
    for dst_ind in range(num_dst):
      latent_code_dst = dst_latent_codes[dst_ind:dst_ind + 1]
      sess.run(setter, {latent_src: latent_code_src})
      dst_imgs = dst_images_orin[dst_ind:dst_ind + 1]
      visualizer.set_cell(src_ind*num_dst+dst_ind, 0, text=src_images_name[src_ind])
      visualizer.set_cell(src_ind*num_dst+dst_ind, 1, image=img_src[0])
      visualizer.set_cell(src_ind*num_dst+dst_ind, 2, image=src_images_invi[src_ind])
      col_idx = 3
      for it in range(1, args.num_iterations+1):
        output_node = [train_op, loss, loss_feat, loss_pixel, loss_adv, w_loss]
        feed_dict = {x: dst_imgs, latent_dst: latent_code_dst}
        _, loss_, feat_loss_, loss_pixel_, loss_adv_, w_loss_ = sess.run(output_node, feed_dict)
        if it % save_interval == 0:
          x_rec_ = sess.run(x_rec)
          imgs_ = adjust_pixel_range(x_rec_)
          visualizer.set_cell(src_ind*num_dst+dst_ind, col_idx, image=imgs_[0])
          col_idx += 1
          print(f'Iter: {it:04d} loss: {np.mean(loss_):6.4f} feat_loss: {np.mean(feat_loss_):6.4f}'
                f' pixel_loss: {np.mean(loss_pixel_):6.4f}  adv: {np.mean(loss_adv_):6.4f} '
                f'w_loss: {np.mean(w_loss_):6.4}')
      visualizer.set_cell(src_ind * num_dst + dst_ind, col_idx, image=dst_images_invi[dst_ind])
  visualizer.save(f'{output_dir}/{job_name}.html')


if __name__ == '__main__':
  main()
