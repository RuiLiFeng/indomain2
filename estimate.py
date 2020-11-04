import os
import argparse
import pickle
from training.misc import progress as tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

from perceptual_model import PerceptualModel
from training import misc


batch_size = 512
num_sample_w = 1024000
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
tflib.init_tf()

_, _, Gs = misc.load_pkl('/home/admin/workspace/project/data/indomain2/pkl/indomain_pkl/stylegan_mod_ffhq256.pkl')

latent_dim = Gs.components.mapping.input_shape[1]
print('Input latent of mapping layer: [%d, %d] tensors' % (batch_size, latent_dim))
Z = tf.random.normal([batch_size, latent_dim])
W = Gs.components.mapping.get_output_for(Z, None)
sess = tf.get_default_session()
ws = []
print('Sample %d Ws with %d batch size and %d batch' %
      (num_sample_w, batch_size, num_sample_w // batch_size))
for batch in tqdm(range(num_sample_w // batch_size)):
    w_batch = sess.run(W)
    ws.append(np.reshape(w_batch, [batch_size, -1]))

ws = np.concatenate(ws, axis=0) # [num_w, w_dim]

w_avg = np.mean(ws, axis=0, keepdims=True)

w_cov = np.cov(np.transpose(ws))

w_e, w_v = np.linalg.eigh(w_cov)


def elipse(w):
    w_transform = np.matmul(w - w_avg, w_v) / np.sqrt(w_e)
    return np.sum(w_transform @ np.transpose(w_transform), axis=1)


def stable_elipse(w, epsilon, radius, return_prop=False):
    mask = (w_e > epsilon) * 1.0
    w_se = mask * w_e + (1 - mask) * epsilon
    w_transform = np.matmul(w - w_avg, w_v) / np.sqrt(w_se * radius)
    if return_prop:
        return np.mean(np.sum(w_transform * w_transform, axis=1) > 1.0)
    return np.max(w_transform * w_transform, axis=1)


w_test = sess.run(W)
score = stable_elipse(np.reshape(w_test, [batch_size, -1]), 0, 2, True)

def trunc(w, t_psi, t_ctf, return_stable):
    ceofs = np.where(w_e > t_ctf, w_e, t_ctf)
    eighs = (w - w_avg) @ w_v / np.sqrt(ceofs * t_psi)
    norm = np.sqrt(np.sum(eighs * eighs, axis=1, keepdims=True))
    norm = np.tile(norm, [1, w.shape[1]])
    w_p = np.where(norm > 1.0, eighs / norm, eighs)
    w_r = w_p @ np.transpose(w_v) * np.sqrt(ceofs * t_psi)
    if return_stable:
        return stable_elipse(w, t_ctf, t_psi, True), stable_elipse(w_r + w_avg, t_ctf, t_psi, True)
    return w_r + w_avg


def square_trunc(w, t_psi, t_ctf, return_stable):
    ceofs = np.where(w_e > t_ctf, w_e, t_ctf)
    eighs = (w - w_avg) @ w_v / np.sqrt(ceofs * t_psi)
    w_p = np.where(eighs > 1.0, 1.0, eighs)
    w_p = np.where(w_p < -1.0, -1.0, w_p)
    w_r = w_p @ np.transpose(w_v) * np.sqrt(ceofs * t_psi)
    if return_stable:
        return stable_elipse(w, t_ctf, t_psi, True), stable_elipse(w_r + w_avg, t_ctf, t_psi, True)
    return w_r + w_avg