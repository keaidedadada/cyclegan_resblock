import tensorflow as tf
import matplotlib.pyplot as plt

import datarecoder
import numpy as np
import math
import cv2
import discriminator_v1 as discriminator
import generator_v1 as generator
from absl.flags import FLAGS
from absl import app, flags


# flags.DEFINE_boolean('use_lsgan', True, 'use lsgan (mean squared error) or cross entropy loss, default: True')
# flags.DEFINE_integer('batch_size', 1, 'The batch_size for training.')
# flags.DEFINE_string('norm', 'instance', '[instance, batch] use instance norm or batch norm, default: instance')
# flags.DEFINE_integer('lambda1', 10,
#                      'weight for forward cycle loss (X->Y->X), default: 10')
# flags.DEFINE_integer('lambda2', 10,
#                      'weight for backward cycle loss (Y->X->Y), default: 10')
# flags.DEFINE_float('learning_rate', 2e-4,
#                    'initial learning rate for Adam, default: 0.0002')
# flags.DEFINE_float('beta1', 0.5,
#                    'momentum term of Adam, default: 0.5')
# flags.DEFINE_integer('ngf', 32,
#                      'number of gen filters in first conv layer, default: 64')
# flags.DEFINE_integer('ndf', 64,
#                      'number of dis filters in first conv layer, default: 32')
# flags.DEFINE_float('weight_decay', 0, "The weight decay for ENet convolution layers.")


def my_read_func(image_path):
    # ima_path = image_path.decode('utf-8')
    img = cv2.imread(image_path, -1)
    img = cv2.resize(img, (256, 256))
    img = (np.array(img, dtype='float32') / 1000) * 2 - 1.
    img = np.expand_dims(img, axis=2)
    img = img[np.newaxis, :, :, :]
    img = tf.convert_to_tensor(img)
    return img


def test():

    log_dir = 'log/resnet/ckpt-145'
    ndf = 64
    norm = 'instance'
    ngf = 32
    G = generator.Generator('G', ngf, norm=norm)
    F = generator.Generator('F', ngf, norm=norm)
    D_Y = discriminator.Discriminator('D_Y', ndf, norm=norm)
    D_X = discriminator.Discriminator('D_X', ndf, norm=norm)
    forbuild = np.random.rand(1, 256, 256, 1).astype(np.float32)
    built = G(forbuild)
    built = F(forbuild)
    built = D_Y(forbuild)
    built = D_X(forbuild)

    # trainA_set = datarecoder.load_dataset(["testA.tfrecords"])
    # trainB_set = datarecoder.load_dataset(["testB.tfrecords"])
    # iteratorA = tf.compat.v1.data.make_one_shot_iterator(trainA_set)
    # iteratorB = tf.compat.v1.data.make_one_shot_iterator(trainB_set)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    ckpt = tf.train.Checkpoint(G=G, F=F, D_X=D_X, D_Y=D_Y, generator_g_optimizer=generator_g_optimizer,
                               generator_f_optimizer=generator_f_optimizer,
                               discriminator_x_optimizer=discriminator_x_optimizer,
                               discriminator_y_optimizer=discriminator_y_optimizer)

    sourcepath = r'D:\yanyi\codes\CycleGAN-master\dataset\r2o\testA\512_2x_2048_63_grey.tif'
    targetpath = r'D:\yanyi\codes\CycleGAN-master\dataset\r2o\testB\5x_basic_1024_46_dgrey2.tif'

    source = my_read_func(sourcepath)
    target = my_read_func(targetpath)

    ckpt.restore(log_dir).expect_partial()

    pred = G(source)
    cycled_x = F(pred)

    fake_x = F(target)
    cycled_y = G(fake_x)

    same_x = F(source)
    same_y = G(target)

    gen_imgs = np.concatenate([source, pred, cycled_x, same_x, target, fake_x, cycled_y, same_y]).squeeze()

    # Rescale images 0 - 1
    gen_imgs = (gen_imgs + 1) / 2

    titles = ['Original', 'Translated', 'Reconstructed', 'Same']
    fig, axs = plt.subplots(2, 4)
    cnt = 0
    for i in range(2):
        for j in range(4):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()
    plt.close()
    p16img = np.array(pred).squeeze()
    p16img1 = np.uint16(((p16img + 1) / 2) * 1000)
    s16img = np.uint16(((np.array(source).squeeze()+1)/2) * 1000)
    cv2.imwrite(r'log\resnet\pred3.tif', p16img1)
    cv2.imwrite(r'log\resnet\source3.tif', s16img)


if __name__ == '__main__':
    import os, sys

    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test()

# # 如果tensorflow是2.0以上版本的话，
# import tensorflow as tf
#
# # 要看的tensorflow模型路径，指向tensorflow模型的saved_model文件夹
# export_dir = ".//saved_model"

# # 把所有节点打印出来
# with tf.Session(graph=tf.Graph()) as sess:
#     tf.saved_model.loader.load(sess, ['serve'], export_dir)
#     print('load model done.')
#     input_graph_def = tf.get_default_graph().as_graph_def()
#     node_names = [n.name for n  in input_graph_def.node]
#     for node in node_names:
#         print(node)


# fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(8, 8))
# axs[0].imshow((imgs_A[0, :, :, 0] * 0.5 + 0.5))
# axs[1].imshow(predictedimg)
# plt.show()

# # 查看cpkt模型参数
# tf.compat.v1.disable_v2_behavior()
# checkpoint_path = r'D:\yanyi\codes\CycleGAN-master\log\nofog\ckpt-75'
# # Read data from checkpoint file
# reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# # Print tensor name and values
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     # print(reader.get_tensor(key))
