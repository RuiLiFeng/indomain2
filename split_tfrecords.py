import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import argparse
import sys


def session(graph=None, allow_soft_placement=True,
            log_device_placement=False, allow_growth=True):
    """ return a Session with simple config """

    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)

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


def get_images(data_dir, sess, batch_size):
    dset = tf.data.TFRecordDataset(data_dir)
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=16)
    dset = dset.batch(batch_size)
    train_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
    training_init_op =train_iterator.make_initializer(dset)
    image_batch = train_iterator.get_next()
    sess.run(training_init_op)
    return image_batch


def main(args):
    print(args)

    sess = session()
    image_batch = get_images(data_dir=args.data_path, sess=sess, batch_size=args.batch_size)

    start = args.start
    end   = args.end
    tfr_prefix = args.save_dir
    os.makedirs(tfr_prefix, exist_ok=True)

    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
    tfr_file = tfr_prefix + '/' + args.save_file_name
    tfr_writer = tf.python_io.TFRecordWriter(tfr_file, tfr_opt)

    for i in tqdm(range(args.total_nums)):
        img = sess.run(image_batch)
        if i >= start and i < end:
            img_ = img[0, :, :, :]
            quant = np.rint(img_).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())

    tfr_writer.close()


if __name__ == "__main__":

    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Location of the tfrecords images')
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Location to save tfrecords file")
    parser.add_argument("--save_file_name", type=str, required=True,
                        help="save file name")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="size of the input batch")
    parser.add_argument('--start', type=int, default=65000)
    parser.add_argument('--end', type=int, default=70000)
    parser.add_argument('--total_nums', type=int, default=70000)

    args = parser.parse_args()
    main(args)