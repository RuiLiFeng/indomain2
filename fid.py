# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import tensorflow as tf

import numpy as np
from cal_metrix import misc
from cal_metrix import tfutil
import time
import cv2
import glob
# from tqdm import tqdm
import time
import sys


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



def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False, shrink=1):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if shrink > 1:
        ksize = [1, 1, shrink, shrink]
        images = tf.nn.avg_pool(images, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW")
    if nchw_to_nhwc:
        images = tf.transpose(images, [0, 2, 3, 1])
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    return tf.saturate_cast(images, tf.uint8)


def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def session(graph=None, allow_soft_placement=True,
            log_device_placement=False, allow_growth=True):
    """ return a Session with simple config """

    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)


def get_train_data(sess, data_dir, batch_size):
    dset = tf.data.TFRecordDataset(data_dir)
    dset = dset.map(parse_tfrecord_tf)
    dset = dset.shuffle(((4096 << 20) - 1) // np.prod([64,64,3]) + 1).batch(batch_size)
    train_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
    training_init_op = train_iterator.make_initializer(dset)
    image_batch = train_iterator.get_next()
    sess.run(training_init_op)
    return image_batch


def evaluate_metrics(log, metrics, num_images, real_passes, image_shape, minibatch_size=64):
    import cal_metrix.metrics.frechet_inception_distance
    import cal_metrix.metrics.sliced_wasserstein
    import cal_metrix.metrics.inception_score
    import cal_metrix.metrics.ms_ssim
    metric_class_names = {
        'swd':      'cal_metrix.metrics.sliced_wasserstein.API',
        'fid':      'cal_metrix.metrics.frechet_inception_distance.API',
        'is':       'cal_metrix.metrics.inception_score.API',
        'msssim':   'cal_metrix.metrics.ms_ssim.API',
    }

    result_subdir = 'results'
    log_file = os.path.join(result_subdir, log)
    print('Logging output to', log_file)
    misc.set_output_log_file(log_file)

    mirror_augment = True

    # Initialize metrics.
    metric_objs = []
    for name in metrics:
        class_name = metric_class_names.get(name, name)
        print('Initializing %s...' % class_name)
        class_def = tfutil.import_obj(class_name)
        image_shape = [3] + image_shape[1:]
        obj = class_def(num_images=num_images, image_shape=image_shape, image_dtype=np.uint8, minibatch_size=minibatch_size)
        #tfutil.init_uninited_vars()
        mode = 'warmup'
        obj.begin(mode)
        for idx in range(10):
            obj.feed(mode, np.random.randint(0, 256, size=[minibatch_size]+image_shape, dtype=np.uint8))
        obj.end(mode)
        metric_objs.append(obj)

    # Print table header.
    print()
    print('%-15s%-12s' % ('Snapshot', 'Time_eval'), end='')
    for obj in metric_objs:
        for name, fmt in zip(obj.get_metric_names(), obj.get_metric_formatting()):
            print('%-*s' % (len(fmt % 0), name), end='')
    print()
    print('%-15s%-12s' % ('---', '---'), end='')
    for obj in metric_objs:
        for fmt in obj.get_metric_formatting():
            print('%-*s' % (len(fmt % 0), '---'), end='')
    print()


    loops = num_images // minibatch_size
    print('Total loops of image: {}'.format(loops))

    Real = glob.glob('/gdata2/fengrl/inverted/ffhq-256-500img/*.png')
    print('Real number of image :', len(Real))
    Real = np.random.permutation(Real)

    for title, mode in [('Reals', 'reals'), ('Reals2', 'fakes')][:real_passes]:
        print('%-15s' % title, end='')
        time_begin = time.time()
        [obj.begin(mode) for obj in metric_objs]
        for ind in tqdm(range(0, num_images, minibatch_size)):
            batch = []
            for i in range(minibatch_size):
                image = cv2.imread(Real[ind+i])
                image = image[:, :, ::-1]
                batch.append(image)
            batch_images = np.asarray(batch)
            images = batch_images.transpose(0, 3, 1, 2)

            if mirror_augment:
                images = misc.apply_mirror_augment(images)
            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1]) # grayscale => RGB
            [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()


    data_dir = ''
    sub_dir = ['/gdata2/fengrl/inverted/indomain2-proj-500/',
               '/gdata2/fengrl/inverted/indomain2-500/',
               '/gdata2/fengrl/inverted/indomain2-r2-500/',
               '/gdata2/fengrl/inverted/2trunc-proj-500/']

    for dir in sub_dir:
        img_dir = data_dir + dir + '/*.png'
        Images = glob.glob(img_dir)[:500]
        print('number of image in {}'.format(len(Images)))
        Images = np.random.permutation(Images)
        tlt = dir
        print('%-20s' % tlt, end='')
        mode ='fakes'
        [obj.begin(mode) for obj in metric_objs]
        time_begin = time.time()
        for ind in tqdm(range(0, num_images, minibatch_size)):
            batch = []
            for i in range(minibatch_size):
                image = cv2.imread(Images[ind + i])
                image = image[:, :, ::-1]
                batch.append(image)
            batch_images = np.asarray(batch)
            gen_images = batch_images.transpose(0, 3, 1, 2)

            if mirror_augment:
                gen_images = misc.apply_mirror_augment(gen_images)
            if gen_images.shape[1] == 1:
                gen_images = np.tile(gen_images, [1, 3, 1, 1]) # grayscale => RGB
            [obj.feed(mode, gen_images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()


if __name__ == "__main__":

    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--batch_size", type=int,
                        default=10, help="Image size")

    hps = parser.parse_args()

    log = 'metric-fid-16k.txt'
    metrics = ['fid']

    sess = session()
    sess.run(tf.global_variables_initializer())
    im_shape = [3,256,256]
    num_imgs = 500
    evaluate_metrics(log=log, metrics=metrics, num_images=num_imgs, real_passes=1, image_shape=im_shape, minibatch_size=hps.batch_size)
