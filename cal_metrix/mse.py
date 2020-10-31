import numpy as np
import tensorflow as tf
import os
from training.misc import progress as tqdm
import argparse
import cv2
import glob


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def tensorflow_session():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.85
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def main(hps):
    with tf.name_scope('input'):
        real = tf.placeholder('float32', [hps.batch_size, 3, hps.image_size, hps.image_size], name='real_image')
        fake = tf.placeholder('float32', [hps.batch_size, 3, hps.image_size, hps.image_size], name='fake_image')
    loss2 = tf.reduce_mean(tf.square(real - fake))
    sess = tensorflow_session()

    data_dir = ''
    sub_dir = ['/gdata2/fengrl/inverted/indomain2-500/', '/gdata2/fengrl/inverted/indomain2-r2-500/']

    loss_list = []
    real_images = sorted(glob.glob('/gdata2/fengrl/inverted/ffhq-256-500img/*.png'))
    num_images = len(real_images)
    print('real image: ', num_images)
    for dir in tqdm(sub_dir):
        img_dir = os.path.join(data_dir, dir)
        fake_images = sorted(glob.glob(img_dir + '/*_inv.png'))
        print('fake image: ', len(fake_images))
        total_loss = 0
        for ind in tqdm(range(0, num_images, hps.batch_size)):
            batch_real = []
            for i in range(hps.batch_size):
                image = cv2.imread(real_images[ind + i])
                image = image[:, :, ::-1]
                batch_real.append(image)
            batch_images = np.asarray(batch_real)
            real_data = batch_images.transpose(0, 3, 1, 2)
            real_data = adjust_dynamic_range(real_data, [0, 255], [-1., 1.])

            batch_fake = []
            for i in range(hps.batch_size):
                image = cv2.imread(fake_images[ind + i])
                image = image[:, :, ::-1]
                batch_fake.append(image)
            batch_images = np.asarray(batch_fake)
            fake_data = batch_images.transpose(0, 3, 1, 2)
            fake_data = adjust_dynamic_range(fake_data, [0, 255], [-1., 1.])

            feed_dict = {real: real_data, fake:fake_data}
            loss2_ = sess.run(loss2, feed_dict)
            total_loss += loss2_
        mse_loss = total_loss / (num_images/hps.batch_size)
        loss_list.append(mse_loss)
    for it in range(len(sub_dir)):
        print(sub_dir[it])
        print(loss_list[it])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10, help="")
    parser.add_argument("--image_size", type=int, default=256, help="")

    hps = parser.parse_args()

    main(hps)
