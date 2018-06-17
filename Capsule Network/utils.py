import os
import scipy
import numpy as np
import tensorflow as tf


def load_emnist(batch_size, is_training=True):
    path = os.path.join('data', 'emnist')
    if is_training:
        fd = open(os.path.join(path, 'emnist-byclass-train-images-idx3-ubyte_digits_only'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        number_of_files = int(len(loaded[16:]) / (28 * 28))
        trainX = loaded[16:].reshape((number_of_files, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'emnist-byclass-train-labels-idx1-ubyte_digits_only'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((number_of_files)).astype(np.int32)

        train_ratio = 0.8
        num_train = int(number_of_files * train_ratio)

        trX = trainX[:num_train] / 255.
        trY = trainY[:num_train]

        valX = trainX[num_train:, ] / 255.
        valY = trainY[num_train:]

        num_tr_batch = num_train // batch_size
        num_val_batch = (number_of_files - num_train) // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 'emnist-byclass-test-images-idx3-ubyte_digits_only'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        number_of_test = len(loaded[16:]) / (28 * 28)
        teX = loaded[16:].reshape((number_of_test, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 'emnist-byclass-test-labels-idx1-ubyte_digits_only'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((number_of_test)).astype(np.int32)

        num_te_batch = number_of_test // batch_size
        return teX / 255., teY, num_te_batch


def load_overlapping_emnist(batch_size, i, is_training=True):
    path = os.path.join('data', 'multi_emnist')
    if is_training:
        fd = open(os.path.join(path, 'emnist-byclass-train-images-idx3-ubyte_overlap_2_0.{}0'.format(i)))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        number_of_files = int(len(loaded[16:])/(28*28))
        trainX = loaded[16:].reshape((number_of_files, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'emnist-byclass-train-labels-idx1-ubyte_overlap_2_0.{}0'.format(i)))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((number_of_files)).astype(np.int32)

        # train_ratio = 0.8
        # num_train = int(number_of_files * train_ratio)

        # trX = trainX / 255.
        # trY = trainY

        # valX = trainX[num_train:, ] / 255.
        # valY = trainY[num_train:]

        # num_tr_batch = num_train // batch_size
        # num_val_batch = (number_of_files - num_train) // batch_size

        return trainX / 255., trainY, number_of_files // batch_size
    else:
        fd = open(os.path.join(path, 'emnist-byclass-test-images-idx3-ubyte_overlap_0_{}0'.format(i)))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        number_of_test = int(len(loaded[16:])/(28*28))
        teX = loaded[16:].reshape((number_of_test, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 'emnist-byclass-test-labels-idx1-ubyte_overlap_0_{}0'.format(i)))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((number_of_test)).astype(np.int32)

        num_te_batch = number_of_test // batch_size
        return teX / 255., teY, num_te_batch


def load_raw_images(path):
    from scipy import misc
    teX = np.array([np.rot90(np.fliplr(misc.imread(os.path.join(path, f)))) for f in os.listdir(path) if f.endswith(".png")])
    teX = teX.reshape((*teX.shape, 1)).astype(np.float)

    return teX


def load_data(dataset, batch_size, i, is_training=True):
    if dataset == 'emnist':
        return load_emnist(batch_size, is_training)
    elif dataset == 'emnist_overlap':
        return load_overlapping_emnist(batch_size, is_training, i)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads, i):
    if dataset == 'emnist':
        trX, trY, tr_batch_num = load_emnist(batch_size, is_training=True)
    elif dataset == 'emnist_overlap':
        trX, trY, tr_batch_num = load_overlapping_emnist(batch_size, i, is_training=True)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y, tr_batch_num)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)
