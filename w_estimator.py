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
  parser.add_argument('--max_values', type=float, default=100,
                      help='The max score values for optimization')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--reverse', action='store_true',
                      help='Decide which direction to optimize')
  parser.add_argument('--num_images', type=int, default=10)
  parser.add_argument('--model_name', type=str, default='ffhq')
  parser.add_argument('--attr_name', type=str, default='expression')
  return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    _, _, Gs = misc.load_pkl(args.model_path)

    latent_dim = Gs.components.mapping.input_shape[1]
    print('Input latent of mapping layer: [%d, %d] tensors' % (args.batch_size, latent_dim))

    ws = []
    print('Sample %d Ws with %d batch size and %d batch' %
          (args.num_sample_w, args.batch_size, args.num_sample_w // args.batch_size))
    for batch in tqdm(range(args.num_sample_w // args.batch_size)):
        z = tf.random.normal([args.batch_size, latent_dim])
        w_batch = Gs.components.mapping.get_output_for(z, None)
        ws.append(tf.reshape(w_batch, [args.batch_size, -1]))
    ws = tf.concat(ws, axis=0) # [num_w, w_dim]

    w_avg = tf.reduce_mean(ws, axis=0, keepdims=True)

    w_cov = tf.matmul(tf.transpose(ws - w_avg), ws - w_avg) / [7168] # [w_dim, w_dim]

    w_e, w_v = tf.linalg.eigh(w_cov)

    def elipse(w):
        w_transform = tf.matmul(w, w_v) * tf.rsqrt(w_e)
        return tf.reduce_sum(w_transform @ tf.transpose(w_transform), axis=1)


    sess = tf.get_default_session()

    Z = tf.random.normal([args.batch_size, latent_dim])
    W = Gs.components.mapping.get_output_for(Z, None)
    score = elipse(tf.reshape(W, [args.batch_size, -1]))
    sess.run(score)