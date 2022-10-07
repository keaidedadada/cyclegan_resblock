import tensorflow as tf
import tensorflow.keras as keras
import ops_v1 as ops
import numpy as np
import matplotlib.pyplot as plt


class Generator(keras.Model):
    def __init__(self, scope: str = "Generator", ngf: int = 64, reg: float = 0.0005, norm: str = "instance",
                 more: bool = True):
        super(Generator, self).__init__(name=scope)
        self.c7s1_32 = ops.c7s1_k(scope="c7s1_32", k=ngf, reg=reg, norm=norm)
        self.d64 = ops.dk(scope="d64", k=2 * ngf, reg=reg, norm=norm)
        self.d128 = ops.dk(scope="d128", k=4 * ngf, reg=reg, norm=norm)
        if more:
            self.res_output = ops.n_res_blocks(scope="8_res_blocks", n=8, k=4 * ngf, reg=reg, norm=norm)
        else:
            self.res_output = ops.n_res_blocks(scope="6_res_blocks", n=6, k=4 * ngf, reg=reg, norm=norm)
        self.u64 = ops.uk(scope="u64", k=2 * ngf, reg=reg, norm=norm)
        self.u32 = ops.uk(scope="u32", k=ngf, reg=reg, norm=norm)
        self.outconv = ops.c7s1_k(scope="output", k=1, reg=reg, norm=norm)

    def call(self, x, training=False):
        x = self.c7s1_32(x, training=training, activation='Relu')
        # d4p = x.numpy().squeeze()
        # d4p = np.uint16(((d4p + 1) / 2) * 1000)
        # fig, axs = plt.subplots(4, 4)
        # cnt = 0
        # for i in range(4):
        #     for j in range(4):
        #         axs[i, j].imshow(d4p[:, :, cnt])
        #         cnt += 1
        #         axs[i, j].axis('off')
        # plt.show()
        # plt.close()

        x = self.d64(x, training=training)
        # d4p = x.numpy().squeeze()
        # d4p = np.uint16(((d4p + 1) / 2) * 1000)
        # fig, axs = plt.subplots(4, 4)
        # cnt = 0
        # for i in range(4):
        #     for j in range(4):
        #         axs[i, j].imshow(d4p[:, :, cnt])
        #         cnt += 1
        #         axs[i, j].axis('off')
        # plt.show()
        # plt.close()

        x = self.d128(x, training=training)
        # d4p = x.numpy().squeeze()
        # d4p = np.uint16(((d4p + 1) / 2) * 1000)
        # fig, axs = plt.subplots(4, 4)
        # cnt = 0
        # for i in range(4):
        #     for j in range(4):
        #         axs[i, j].imshow(d4p[:, :, cnt])
        #         cnt += 1
        #         axs[i, j].axis('off')
        # plt.show()
        # plt.close()

        x = self.res_output(x, training=training)
        # d4p = x.numpy().squeeze()
        # shape3 = np.shape(d4p)
        # channels = int(shape3[2]/4)
        # d4p = np.uint16(((d4p + 1) / 2) * 1000)
        # fig, axs = plt.subplots(4, channels)
        # cnt = 0
        # for i in range(4):
        #     for j in range(channels):
        #         axs[i, j].imshow(d4p[:, :, cnt])
        #         cnt += 1
        #         axs[i, j].axis('off')
        # plt.show()
        # plt.close()

        x = self.u64(x, training=training)
        # d4p = x.numpy().squeeze()
        # d4p = np.uint16(((d4p + 1) / 2) * 1000)
        # shape3 = np.shape(d4p)
        # channels = int(shape3[2]/4)
        # fig, axs = plt.subplots(4, channels)
        # cnt = 0
        # for i in range(4):
        #     for j in range(channels):
        #         axs[i, j].imshow(d4p[:, :, cnt])
        #         cnt += 1
        #         axs[i, j].axis('off')
        # plt.show()
        # plt.close()

        x = self.u32(x, training=training)
        # d4p = x.numpy().squeeze()
        # d4p = np.uint16(((d4p + 1) / 2) * 1000)
        # shape3 = np.shape(d4p)
        # channels = int(shape3[2]/4)
        # fig, axs = plt.subplots(4, channels)
        # cnt = 0
        # for i in range(4):
        #     for j in range(channels):
        #         axs[i, j].imshow(d4p[:, :, cnt])
        #         cnt += 1
        #         axs[i, j].axis('off')
        # plt.show()
        # plt.close()

        x = self.outconv(x, training=training, activation='tanh')
        # d4p = x.numpy().squeeze()
        # d4p = np.uint16(((d4p + 1) / 2) * 1000)
        # plt.imshow(d4p)
        # plt.show()
        # plt.close()

        return x
