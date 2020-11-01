import numpy as np
import tensorflow as tf
import os
# from training.misc import progress as tqdm
import time
import argparse
import cv2
import glob
import sys
import re


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
    sub_dir = ['/gdata2/fengrl/inverted/indomain2-r0.5-500/',
               '/gdata2/fengrl/inverted/indomain-500/',
               '/gdata2/fengrl/inverted/indomain2-proj-500/',
               '/gdata2/fengrl/inverted/indomain2-500/',
               '/gdata2/fengrl/inverted/indomain2-r2-500/',
               '/gdata2/fengrl/inverted/2trunc-proj-500/']

    loss_list = []
    real_images = sorted(glob.glob('/gdata2/fengrl/inverted/ffhq-256-500img/*.png'), key=lambda x:int(re.findall('\d+',x)[-1]))
    num_images = len(real_images)
    print('real image: ', num_images)
    for dir in tqdm(sub_dir):
        img_dir = os.path.join(data_dir, dir)
        fake_images = sorted(glob.glob(img_dir + '/*_inv.png'), key=lambda x:int(re.findall('\d+',x)[-1]))
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


def tqdm(items, desc='', total=None, min_delay=0.1, displaytype='eta', **kwargs):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % (
                desc, n + 1, total, n / float(total) * 100), end=" ")
            if n > 0:

                if displaytype == 's1k':  # minutes/seconds for 1000 iters
                    next_1000 = n + (1000 - n % 1000)
                    t_done = t_now - t_start
                    t_1k = t_done / n * next_1000
                    outlist = list(divmod(t_done, 60)) + list(divmod(t_1k - t_done, 60))
                    print("(TE/ET1k: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
                else:  # displaytype == 'eta':
                    t_done = t_now - t_start
                    t_total = t_done / n * total
                    outlist = list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))
                    print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")

            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10, help="")
    parser.add_argument("--image_size", type=int, default=256, help="")

    hps = parser.parse_args()

    main(hps)
