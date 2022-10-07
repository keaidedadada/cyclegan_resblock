import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

dataset_dir = r'dataset\c2o'

image_source = sorted(
    [os.path.join(dataset_dir, 'trainA', file) for file in os.listdir(dataset_dir + r'\trainA') if
     file.endswith('.tif')])
image_targrt = sorted(
    [os.path.join(dataset_dir, "trainB", file) for file in os.listdir(dataset_dir + r'\trainB') if
     file.endswith('.tif')])

test_source = sorted(
    [os.path.join(dataset_dir, 'testA', file) for file in os.listdir(dataset_dir + r'\testA') if
     file.endswith('.tif')])
test_targrt = sorted(
    [os.path.join(dataset_dir, "testB", file) for file in os.listdir(dataset_dir + r'\testB') if
     file.endswith('.tif')])

# 将根据数据的具体格式进行编译，并存储为tfrecords格式文件。
# 其中float格式编译为float_list，字符串、高维array等格式直接编译为bytes_list格式。
def save_tfrecords(data, label: bytes, desfile):
    with tf.io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature={
                    "data": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data[i]).numpy()])),
                    # "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=[512, 512, 1])),
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                }
            )
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


################### TFR数据反编译
def map_func(example):
    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string),
        # 'shape': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, features=feature_description)
    # image_raw = tf.io.parse_tensor(parsed_example['data'], tf.float32)
    x_sample = tf.io.parse_tensor(parsed_example['data'], tf.float32)
    y_sample = parsed_example['label']
    # z_sample = parsed_example['shape']
    return x_sample


################### 加载数据集
def load_dataset(filepaths):
    shuffle_buffer_size = 50
    batch_size = 1

    dataset = tf.data.TFRecordDataset(filepaths)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(map_func=map_func, num_parallel_calls=2)
    dataset = dataset.repeat().batch(batch_size)

    return dataset


def my_numpy_func(image_path):
    # ima_path = image_path.decode('utf-8')
    img = cv2.imread(image_path, -1)
    img = cv2.resize(img, (256, 256))
    img = (np.array(img, dtype='float32') / 1000) * 2 - 1.
    img = np.expand_dims(img, axis=2)
    return img


def decodefortrain(image_path):
    img = tf.data.Dataset.from_tensor_slices(image_path)
    img = img.map(lambda x: tf.numpy_function(func=my_numpy_func, inp=[x], Tout=tf.float32))
    return img


if __name__ == '__main__':
    trainAset = []
    for x in image_source:
        img = my_numpy_func(x)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        trainAset.append(img)


        # trainAset = decodefortrain(image_source)
        # trainBset = decodefortrain(image_targrt)
    save_tfrecords(trainAset, b'trainA', "trainA.tfrecords")

    trainBset = []
    for x in image_targrt:
        img = my_numpy_func(x)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        trainBset.append(img)
    save_tfrecords(trainBset, b'trainB', "trainB.tfrecords")

#     train_set = load_dataset(["trainA.tfrecords"])
#
#     iterator = tf.compat.v1.data.make_one_shot_iterator(train_set)
#
#     train_batch = iterator.get_next()
# 1
    # for image_features in train_set:
    #     img = (image_features[0] + 1) / 2
    #     plt.imshow(img[0, :, :, 0])
    #     plt.show()

