import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import gfile
import pickle
import scipy.misc

from nets import inception_v3
import scipy.misc
slim = tf.contrib.slim

def transform(image, resize_height=36, resize_width=64):
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.
def inverse_transform(images):
    return (images+1.)/2.

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=False):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
              initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
        return tf.matmul(input_, matrix) + bias


def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

class AutoDC:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.g_bn3 = batch_norm(name='g_bn3')

    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        len_video = imgshape[1]
        featsize = 1024
        self.batch_size = len_video
        image = tf.reshape(image, [self.batch_size] + imgshape[-3:])
        with tf.variable_scope("conv") as scope:
            h0 = conv2d(image, self.df_dim, name='d_h0_conv')
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            self.h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(self.h3, [self.batch_size, -1]), featsize, 'd_h3_lin')
#             self.sim = h4[:, :featsize]
#             self.diff = h4[:, featsize:]
#             self.simreshape = tf.reshape(self.sim, (-1, len_video, featsize))
#             m = self.sim.get_shape().as_list()[0]
#             self.simloss = tf.nn.l2_loss(self.simreshape[0] - self.simreshape[1]) / m
#             self.diffloss = tf.nn.l2_loss(tf.matmul(self.sim, tf.transpose(self.diff))) / m / m
#             print(m)
            self.z = h4#self.sim + self.diff
            print(self.z.get_shape())

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                self.z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        self.out = h4#tf.nn.tanh(h4)
        self.reconloss = tf.nn.l2_loss(self.out - image)
        self.loss = self.reconloss# + self.diffloss + self.simloss


class TimeDC:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

    def build(self, image, times):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[0]
        featsize = 1024
        with tf.variable_scope("conv") as scope:
            h0 = conv2d(image, self.df_dim, name='d_h0_conv')
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            self.h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = lrelu(linear(tf.reshape(self.h3, [self.batch_size, -1]), featsize, 'd_h3_lin'))
            h5 = lrelu(linear(h4, featsize, 'd_h4_lin'))
            out = linear(h4, 1, 'd_h5_lin')
#             self.sim = h4[:, :featsize]
#             self.diff = h4[:, featsize:]
#             self.simreshape = tf.reshape(self.sim, (-1, len_video, featsize))
#             m = self.sim.get_shape().as_list()[0]
#             self.simloss = tf.nn.l2_loss(self.simreshape[0] - self.simreshape[1]) / m
#             self.diffloss = tf.nn.l2_loss(tf.matmul(self.sim, tf.transpose(self.diff))) / m / m
#             print(m)
            self.z = out#self.sim + self.diff
            print(self.z.get_shape())


#         self.out = h4#tf.nn.tanh(h4)
#         self.reconloss = tf.nn.l2_loss(self.out - image)
        self.loss = tf.nn.l2_loss(out - times) # + self.diffloss + self.simloss
        self.diffs = out - times

class SubspaceAE:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        len_video = imgshape[1]
        featsize = 1024
        self.batch_size = len_video * 2
        imageorig = image
        image = tf.reshape(image, [self.batch_size] + imgshape[-3:])
        with tf.variable_scope("conv") as scope:
            h0 = conv2d(image, self.df_dim, name='d_h0_conv')
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = lrelu(self.d_bn4(linear(tf.reshape(h3, [self.batch_size, -1]), featsize, 'd_h4_lin')))
            h5 = lrelu(self.d_bn5(linear(h4, featsize, 'd_h5_lin')))
            z = linear(h5, featsize, 'd_hz_lin')
#             self.sim = h4[:, :featsize]
#             self.diff = h4[:, featsize:]
            self.simreshape = tf.reshape(z, (-1, len_video, featsize))
#             m = self.sim.get_shape().as_list()[0]
            self.simloss = tf.nn.l2_loss(self.simreshape[0] - self.simreshape[1])# / len_video
#             self.diffloss = tf.nn.l2_loss(tf.matmul(self.sim, tf.transpose(self.diff))) / m / m
#             print(m)
            self.z = z#self.sim + self.diff
            print(self.z.get_shape())

        outs = []
        for j in range(2):
            with tf.variable_scope("deconv" + str(j)) as scope:
                self.g_bn0 = batch_norm(name='g_bn0')
                self.g_bn1 = batch_norm(name='g_bn1')
                self.g_bn2 = batch_norm(name='g_bn2')
                self.g_bn3 = batch_norm(name='g_bn3')
                self.g_bn4 = batch_norm(name='g_bn4')
                self.g_bn5 = batch_norm(name='g_bn5')
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4, s_h8, s_h16 = \
                    int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
                s_w2, s_w4, s_w8, s_w16 = \
                    int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

                z = self.simreshape[j]
                h5 = lrelu(self.g_bn5(linear(z, featsize, 'g_h5_lin')))
                h4 = lrelu(self.g_bn4(linear(h5, featsize, 'g_h4_lin')))
                print (h4.get_shape())

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    h4, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [len_video, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [len_video, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [len_video, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [len_video, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
                outs.append(h4)

        self.outs = outs#tf.nn.tanh(h4)
        self.reconloss = tf.nn.l2_loss(imageorig - self.outs)
        self.loss = self.reconloss + self.simloss

nclass = 50
class TimeSoftmax:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

    def build(self, image, times):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[0]
        featsize = 1024
        with tf.variable_scope("conv") as scope:
            h0 = conv2d(image, self.df_dim, name='d_h0_conv')
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            self.h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = lrelu(linear(tf.reshape(self.h3, [self.batch_size, -1]), featsize, 'd_h3_lin'))
            h5 = lrelu(linear(h4, featsize//2, 'd_h4_lin'))
            logits = linear(h5, nclass, 'd_h5_lin')
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, times))#self.sim + self.diff
            self.z =  tf.nn.softmax(logits)
            print(self.z.get_shape())

#         self.out = h4#tf.nn.tanh(h4)
#         self.reconloss = tf.nn.l2_loss(self.out - image)
#         self.loss = tf.nn.l2_loss(out - times) / batch_size# + self.diffloss + self.simloss

class TimePred:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

    def build(self, image, times):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[0]
        featsize = 1024
        with tf.variable_scope("conv") as scope:
            h0 = conv2d(image, self.df_dim, name='d_h0_conv')
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            self.h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = lrelu(linear(tf.reshape(self.h3, [self.batch_size, -1]), featsize, 'd_h3_lin'))
            h5 = lrelu(linear(h4, featsize//2, 'd_h4_lin'))
            self.z = linear(h5, 1, 'd_h5_lin')
            self.loss = tf.reduce_mean((self.z - times) ** 2)
            print(self.z.get_shape())


class ReachAE:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[0]
        featsize = 1024
        with tf.variable_scope("conv") as scope:
            h0 = conv2d(image, self.df_dim, name='d_h0_conv')
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = lrelu(self.d_bn4(linear(tf.reshape(h3, [self.batch_size, -1]), featsize, 'd_h4_lin')))
            #h5 = lrelu(self.d_bn5(linear(h4, featsize, 'd_h5_lin')))
            z = linear(h4, featsize, 'd_hz_lin')
#             self.sim = h4[:, :featsize]
#             self.diff = h4[:, featsize:]
#             self.simreshape = tf.reshape(z, (-1, len_video, featsize))
#             m = self.sim.get_shape().as_list()[0]
#             self.simloss = tf.nn.l2_loss(self.simreshape[0] - self.simreshape[1])# / len_video
#             self.diffloss = tf.nn.l2_loss(tf.matmul(self.sim, tf.transpose(self.diff))) / m / m
#             print(m)
#             self.z = self.simreshape#self.sim + self.diff
            self.z = z
            print(self.z.get_shape())

        with tf.variable_scope("deconv") as scope:
            self.g_bn0 = batch_norm(name='g_bn0')
            self.g_bn1 = batch_norm(name='g_bn1')
            self.g_bn2 = batch_norm(name='g_bn2')
            self.g_bn3 = batch_norm(name='g_bn3')
            self.g_bn4 = batch_norm(name='g_bn4')
            self.g_bn5 = batch_norm(name='g_bn5')
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            print(z.get_shape())
            #h5 = lrelu(self.g_bn5(linear(z, featsize, 'g_h5_lin')))
            h4 = lrelu(self.g_bn4(linear(z, featsize, 'g_h4_lin')))
            print(h4.get_shape())

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                h4, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            self.h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(self.h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        self.out = h4#tf.nn.tanh(h4)
        self.reconloss = tf.nn.l2_loss(image - self.out)
        self.loss = self.reconloss# + self.simloss


class ContextAE:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

        with tf.variable_scope("conv_context") as scope:
            h0 = conv2d(contextimg, self.df_dim, name='h0_conv')
            c_bn1 = batch_norm(name='c_bn1')
            c_bn2 = batch_norm(name='c_bn2')
            c_bn3 = batch_norm(name='c_bn3')
            c_bn4 = batch_norm(name='c_bn4')
            h1 = lrelu(c_bn1(conv2d(h0, self.df_dim*2, name='h1_conv')))
            h2 = lrelu(c_bn2(conv2d(h1, self.df_dim*4, name='h2_conv')))
            h3 = lrelu(c_bn3(conv2d(h2, self.df_dim*8, name='h3_conv')))
            h4 = lrelu(c_bn4(linear(tf.reshape(h3, [self.batch_size, -1]), featsize, 'h4_lin')))
            #h5 = lrelu(self.d_bn5(linear(h4, featsize, 'd_h5_lin')))
            z_ctx = linear(h4, featsize, 'hz_lin')

        with tf.variable_scope("conv") as scope:
            h0 = conv2d(inputimg, self.df_dim, name='h0_conv')
            c_bn1 = batch_norm(name='c_bn1')
            c_bn2 = batch_norm(name='c_bn2')
            c_bn3 = batch_norm(name='c_bn3')
            c_bn4 = batch_norm(name='c_bn4')
            h1 = lrelu(c_bn1(conv2d(h0, self.df_dim*2, name='h1_conv')))
            h2 = lrelu(c_bn2(conv2d(h1, self.df_dim*4, name='h2_conv')))
            h3 = lrelu(c_bn3(conv2d(h2, self.df_dim*8, name='h3_conv')))
            h4 = lrelu(c_bn4(linear(tf.reshape(h3, [self.batch_size, -1]), featsize, 'h4_lin')))
            #h5 = lrelu(self.d_bn5(linear(h4, featsize, 'd_h5_lin')))
            z = linear(h4, featsize, 'hz_lin')
#             self.sim = h4[:, :featsize]
#             self.diff = h4[:, featsize:]
#             self.simreshape = tf.reshape(z, (-1, len_video, featsize))
#             m = self.sim.get_shape().as_list()[0]
#             self.simloss = tf.nn.l2_loss(self.simreshape[0] - self.simreshape[1])# / len_video
#             self.diffloss = tf.nn.l2_loss(tf.matmul(self.sim, tf.transpose(self.diff))) / m / m
#             print(m)
#             self.z = self.simreshape#self.sim + self.diff
            self.z = z
            print(self.z.get_shape())

        with tf.variable_scope("deconv") as scope:
            d_bn0 = batch_norm(name='d_bn0')
            d_bn1 = batch_norm(name='d_bn1')
            d_bn2 = batch_norm(name='d_bn2')
            d_bn3 = batch_norm(name='d_bn3')
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            #h5 = lrelu(self.g_bn5(linear(z, featsize, 'g_h5_lin')))
#             h4 = lrelu(d_bn4(linear(z, featsize, 'd_h4_lin')))
#             print(h4.get_shape())

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                tf.concat([z, z_ctx], 1), self.gf_dim*8*s_h16*s_w16, 'd_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = lrelu(d_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1', with_w=True)
            h1 = lrelu(d_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2', with_w=True)
            h2 = lrelu(d_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3', with_w=True)
            h3 = lrelu(d_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='d_h4', with_w=True)

        self.out = h4#tf.nn.tanh(h4)
        self.loss = tf.nn.l2_loss(outputimg - self.out)
#         self.loss = self.reconloss# + self.simloss

class ContextSkipAE:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

        with tf.variable_scope("conv_context") as scope:
            c_bn0 = batch_norm(name='c_bn0')
            c_bn1 = batch_norm(name='c_bn1')
            c_bn2 = batch_norm(name='c_bn2')
            c_bn3 = batch_norm(name='c_bn3')
            c_bn4 = batch_norm(name='c_bn4')
            ctx_h0 = lrelu(c_bn0(conv2d(contextimg, self.df_dim, name='h0_conv')))
            ctx_h1 = lrelu(c_bn1(conv2d(ctx_h0, self.df_dim*2, name='h1_conv')))
            ctx_h2 = lrelu(c_bn2(conv2d(ctx_h1, self.df_dim*4, name='h2_conv')))
            ctx_h3 = lrelu(c_bn3(conv2d(ctx_h2, self.df_dim*8, name='h3_conv')))
            ctx_h4 = lrelu(c_bn4(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin')))
            #h5 = lrelu(self.d_bn5(linear(h4, featsize, 'd_h5_lin')))
            ctx_z = linear(ctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv") as scope:
            c_bn0 = batch_norm(name='c_bn0')
            c_bn1 = batch_norm(name='c_bn1')
            c_bn2 = batch_norm(name='c_bn2')
            c_bn3 = batch_norm(name='c_bn3')
            c_bn4 = batch_norm(name='c_bn4')
            h0 = lrelu(c_bn0(conv2d(inputimg, self.df_dim, name='h0_conv')))
            h1 = lrelu(c_bn1(conv2d(h0, self.df_dim*2, name='h1_conv')))
            h2 = lrelu(c_bn2(conv2d(h1, self.df_dim*4, name='h2_conv')))
            h3 = lrelu(c_bn3(conv2d(h2, self.df_dim*8, name='h3_conv')))
            print(h3.get_shape())
            h4 = lrelu(c_bn4(linear(tf.reshape(h3, [self.batch_size, -1]), featsize, 'h4_lin')))
            #h5 = lrelu(self.d_bn5(linear(h4, featsize, 'd_h5_lin')))
            z = linear(h4, featsize, 'hz_lin')
#             self.sim = h4[:, :featsize]
#             self.diff = h4[:, featsize:]
#             self.simreshape = tf.reshape(z, (-1, len_video, featsize))
#             m = self.sim.get_shape().as_list()[0]
#             self.simloss = tf.nn.l2_loss(self.simreshape[0] - self.simreshape[1])# / len_video
#             self.diffloss = tf.nn.l2_loss(tf.matmul(self.sim, tf.transpose(self.diff))) / m / m
#             print(m)
#             self.z = self.simreshape#self.sim + self.diff
            self.z = z
            print(self.z.get_shape())

        with tf.variable_scope("deconv") as scope:
            d_bn0 = batch_norm(name='d_bn0')
            d_bn1 = batch_norm(name='d_bn1')
            d_bn2 = batch_norm(name='d_bn2')
            d_bn3 = batch_norm(name='d_bn3')
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            #h5 = lrelu(self.g_bn5(linear(z, featsize, 'g_h5_lin')))
#             h4 = lrelu(d_bn4(linear(z, featsize, 'd_h4_lin')))
#             print(h4.get_shape())

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                tf.concat([z, ctx_z], 1), self.gf_dim*8*s_h16*s_w16, 'd_h0_lin', with_w=True)

            h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = lrelu(d_bn0(h0))

            h1, self.h1_w, self.h1_b = deconv2d(
                tf.concat([h0, ctx_h3], 3), [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1', with_w=True)
            h1 = lrelu(d_bn1(h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                tf.concat([h1, ctx_h2], 3), [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2', with_w=True)
            h2 = lrelu(d_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                tf.concat([h2, ctx_h1], 3), [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3', with_w=True)
            h3 = lrelu(d_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                tf.concat([h3, ctx_h0], 3), [self.batch_size, s_h, s_w, self.c_dim], name='d_h4', with_w=True)

        self.out = h4#tf.nn.tanh(h4)
        self.loss = tf.nn.l2_loss(outputimg - self.out)
#         self.loss = self.reconloss# + self.simloss

class ContextNoBNAE:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

        with tf.variable_scope("conv_context") as scope:
#             c_bn0 = batch_norm(name='c_bn0')
#             c_bn1 = batch_norm(name='c_bn1')
#             c_bn2 = batch_norm(name='c_bn2')
#             c_bn3 = batch_norm(name='c_bn3')
#             c_bn4 = batch_norm(name='c_bn4')
            ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
            ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
            ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
            ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
            ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            #h5 = lrelu(self.d_bn5(linear(h4, featsize, 'd_h5_lin')))
            ctx_z = linear(ctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv") as scope:
#             c_bn0 = batch_norm(name='c_bn0')
#             c_bn1 = batch_norm(name='c_bn1')
#             c_bn2 = batch_norm(name='c_bn2')
#             c_bn3 = batch_norm(name='c_bn3')
#             c_bn4 = batch_norm(name='c_bn4')
            h0 = lrelu(conv2d(inputimg, self.df_dim, name='h0_conv'))
            h1 = lrelu(conv2d(h0, self.df_dim*2, name='h1_conv'))
            h2 = lrelu(conv2d(h1, self.df_dim*4, name='h2_conv'))
            h3 = lrelu(conv2d(h2, self.df_dim*8, name='h3_conv'))
            print(h3.get_shape())
            h4 = lrelu(linear(tf.reshape(h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            #h5 = lrelu(self.d_bn5(linear(h4, featsize, 'd_h5_lin')))
            z = linear(h4, featsize, 'hz_lin')
#             self.sim = h4[:, :featsize]
#             self.diff = h4[:, featsize:]
#             self.simreshape = tf.reshape(z, (-1, len_video, featsize))
#             m = self.sim.get_shape().as_list()[0]
#             self.simloss = tf.nn.l2_loss(self.simreshape[0] - self.simreshape[1])# / len_video
#             self.diffloss = tf.nn.l2_loss(tf.matmul(self.sim, tf.transpose(self.diff))) / m / m
#             print(m)
#             self.z = self.simreshape#self.sim + self.diff
            self.z = z
            self.simloss = 0
            for j in range(3):
                self.asdf = z[j*25:(j+1) * 25]- z[(j+1) * 25 : (j+2) * 25]
                self.simloss += tf.reduce_mean((z[j*25:(j+1) * 25]- z[(j+1) * 25 : (j+2) * 25]) ** 2)/3
            mean, var = tf.nn.moments(self.z,axes=[0])
            print(var.get_shape())

            self.simloss /= tf.reduce_mean(var)
            print(self.z.get_shape())

        with tf.variable_scope("deconv") as scope:
#             d_bn0 = batch_norm(name='d_bn0')
#             d_bn1 = batch_norm(name='d_bn1')
#             d_bn2 = batch_norm(name='d_bn2')
#             d_bn3 = batch_norm(name='d_bn3')
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            #h5 = lrelu(self.g_bn5(linear(z, featsize, 'g_h5_lin')))
#             h4 = lrelu(d_bn4(linear(z, featsize, 'd_h4_lin')))
#             print(h4.get_shape())

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                tf.concat([z, ctx_z], 1), self.gf_dim*8*s_h16*s_w16, 'd_h0_lin', with_w=True)

            h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = lrelu(h0)

            h1, self.h1_w, self.h1_b = deconv2d(
                tf.concat([h0, ctx_h3], 3), [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1', with_w=True)
            h1 = lrelu(h1)

            h2, self.h2_w, self.h2_b = deconv2d(
                tf.concat([h1, ctx_h2], 3), [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2', with_w=True)
            h2 = lrelu(h2)

            h3, self.h3_w, self.h3_b = deconv2d(
                tf.concat([h2, ctx_h1], 3), [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3', with_w=True)
            h3 = lrelu(h3)

            h4, self.h4_w, self.h4_b = deconv2d(
                tf.concat([h3, ctx_h0], 3), [self.batch_size, s_h, s_w, self.c_dim], name='d_h4', with_w=True)

        self.out = h4#tf.nn.tanh(h4)
        self.loss = tf.nn.l2_loss(outputimg - self.out) + 1e3 * self.simloss
#         self.loss = self.reconloss# + self.simloss

class ContextAEDeconv:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

        with tf.variable_scope("conv_context") as scope:
            ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
            ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
            ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
            ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
            ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            ctx_z = linear(ctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv_input") as scope:
            input_h0 = lrelu(conv2d(inputimg, self.df_dim, name='h0_conv'))
            input_h1 = lrelu(conv2d(input_h0, self.df_dim*2, name='h1_conv'))
            input_h2 = lrelu(conv2d(input_h1, self.df_dim*4, name='h2_conv'))
            input_h3 = lrelu(conv2d(input_h2, self.df_dim*8, name='h3_conv'))
            print(input_h3.get_shape())
            input_h4 = lrelu(linear(tf.reshape(input_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            input_z = linear(input_h4, featsize, 'hz_lin')
            self.input_z = input_z
            self.simloss = tf.zeros(())
            print(self.input_z.get_shape())

            scope.reuse_variables()

            truth_h0 = lrelu(conv2d(outputimg, self.df_dim, name='h0_conv'))
            truth_h1 = lrelu(conv2d(truth_h0, self.df_dim*2, name='h1_conv'))
            truth_h2 = lrelu(conv2d(truth_h1, self.df_dim*4, name='h2_conv'))
            truth_h3 = lrelu(conv2d(truth_h2, self.df_dim*8, name='h3_conv'))
            print(truth_h3.get_shape())
            truth_h4 = lrelu(linear(tf.reshape(truth_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            truth_z = linear(truth_h4, featsize, 'hz_lin')
            self.truth_z = truth_z
            self.simloss = tf.zeros(())
            print(self.truth_z.get_shape())


        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            self.output_z_ = lrelu(linear(
                tf.concat([input_z, ctx_z], 1), self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(self.output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            self.output_h1 = lrelu(deconv2d(tf.concat([output_h0, ctx_h3], 3),
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            self.output_h2 = lrelu(deconv2d(tf.concat([self.output_h1, ctx_h2], 3),
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            self.output_h3 = lrelu(deconv2d(tf.concat([self.output_h2, ctx_h1], 3),
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(tf.concat([self.output_h3, ctx_h0], 3),
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            scope.reuse_variables()
            truthoutput_z_ = lrelu(linear(
                tf.concat([truth_z, ctx_z], 1), self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(tf.concat([truthoutput_h0, ctx_h3], 3),
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(tf.concat([truthoutput_h1, ctx_h2], 3),
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            print(truthoutput_h2.get_shape())

        self.simloss = tf.nn.l2_loss(truthoutput_h2 - self.output_h2)
        self.out = output_h4#tf.nn.tanh(h4)
        self.loss = tf.nn.l2_loss(outputimg - self.out) + 1e2 * self.simloss
#         self.loss = self.reconloss# + self.simloss

class ContextRes:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

        with tf.variable_scope("conv_context") as scope:
            ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
            ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
            ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
            ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
            ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            ctx_z = linear(ctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv_input") as scope:
            input_h0 = lrelu(conv2d(inputimg, self.df_dim, name='h0_conv'))
            input_h1 = lrelu(conv2d(input_h0, self.df_dim*2, name='h1_conv'))
            input_h2 = lrelu(conv2d(input_h1, self.df_dim*4, name='h2_conv'))
            input_h3 = lrelu(conv2d(input_h2, self.df_dim*8, name='h3_conv'))
            print(input_h3.get_shape())
            input_h4 = lrelu(linear(tf.reshape(input_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            input_z = linear(input_h4, featsize, 'hz_lin')
            self.input_z = input_z
#             self.simloss = tf.zeros(())
            print(self.input_z.get_shape())
            self.simloss = 0
            for j in range(3):
#                 self.asdf = z[j*25:(j+1) * 25]- z[(j+1) * 25 : (j+2) * 25]
                self.simloss += tf.reduce_mean((input_z[j*25:(j+1) * 25]-
                                                input_z[(j+1) * 25 : (j+2) * 25]) ** 2)
            mean, var = tf.nn.moments(input_z,axes=[0])
            print(var.get_shape())
            self.simloss /= tf.reduce_mean(var)

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            self.output_z_ = lrelu(linear(
                tf.concat([input_z, ctx_z], 1), self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(self.output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(output_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            output_h2 = lrelu(deconv2d(output_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            output_h3 = lrelu(deconv2d(output_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(output_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

#         self.simloss = tf.nn.l2_loss(truthoutput_h2 - self.output_h2)
        self.out = output_h4 + contextimg#tf.nn.tanh(h4)
        self.loss = tf.nn.l2_loss(outputimg - self.out) + 1e3 * self.simloss
#         self.loss = self.reconloss# + self.simloss

class ContextResTranslate:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

        with tf.variable_scope("conv_context") as scope:
            ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
            ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
            ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
            ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
            ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            ctx_z = linear(ctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv_input") as scope:
            input_h0 = lrelu(conv2d(inputimg, self.df_dim, name='h0_conv'))
            input_h1 = lrelu(conv2d(input_h0, self.df_dim*2, name='h1_conv'))
            input_h2 = lrelu(conv2d(input_h1, self.df_dim*4, name='h2_conv'))
            input_h3 = lrelu(conv2d(input_h2, self.df_dim*8, name='h3_conv'))
            print(input_h3.get_shape())
            input_h4 = lrelu(linear(tf.reshape(input_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            input_z = lrelu(linear(input_h4, featsize, 'hz_lin'))
            self.input_z = input_z
            input_zh0 = lrelu(linear(tf.concat([input_z, ctx_z], 1), featsize, 'zh0'))
            self.translated_z = linear(input_zh0, featsize, 'translate_z')
#             self.simloss = tf.zeros(())
#             print(self.input_z.get_shape())
#             self.simloss = 0
#             for j in range(3):
# #                 self.asdf = z[j*25:(j+1) * 25]- z[(j+1) * 25 : (j+2) * 25]
#                 self.simloss += tf.reduce_mean((input_z[j*25:(j+1) * 25]-
#                                                 input_z[(j+1) * 25 : (j+2) * 25]) ** 2)
#             mean, var = tf.nn.moments(input_z,axes=[0])
#             print(var.get_shape())
#             self.simloss /= tf.reduce_mean(var)
            scope.reuse_variables()

            truth_h0 = lrelu(conv2d(outputimg, self.df_dim, name='h0_conv'))
            truth_h1 = lrelu(conv2d(truth_h0, self.df_dim*2, name='h1_conv'))
            truth_h2 = lrelu(conv2d(truth_h1, self.df_dim*4, name='h2_conv'))
            truth_h3 = lrelu(conv2d(truth_h2, self.df_dim*8, name='h3_conv'))
            print(truth_h3.get_shape())
            truth_h4 = lrelu(linear(tf.reshape(truth_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            truth_z = lrelu(linear(truth_h4, featsize, 'hz_lin'))
            self.truth_z = truth_z
            self.simloss = tf.zeros(())
            print(self.truth_z.get_shape())

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            self.output_z_ = lrelu(linear(self.translated_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(self.output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(output_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            output_h2 = lrelu(deconv2d(output_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            output_h3 = lrelu(deconv2d(output_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(output_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            scope.reuse_variables()

            truthoutput_z_ = lrelu(linear(truth_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(truthoutput_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(truthoutput_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            truthoutput_h3 = lrelu(deconv2d(truthoutput_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            truthoutput_h4 = deconv2d(truthoutput_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

#         self.simloss = tf.nn.l2_loss(truthoutput_h2 - self.output_h2)
        self.simloss = tf.nn.l2_loss(self.translated_z - self.truth_z)
        self.out = output_h4 + contextimg#tf.nn.tanh(h4)
        self.recon1 = tf.nn.l2_loss(outputimg - self.out)
        self.recon2 = tf.nn.l2_loss(outputimg - (contextimg + truthoutput_h4))
        self.loss = self.recon1 + self.recon2 + 1e3 * self.simloss
#         self.loss = self.reconloss# + self.simloss

class ContextVanilla:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

        with tf.variable_scope("conv_context") as scope:
            ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
            ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
            ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
            ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
            ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            ctx_z = linear(ctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv_input") as scope:
            input_h0 = lrelu(conv2d(inputimg, self.df_dim, name='h0_conv'))
            input_h1 = lrelu(conv2d(input_h0, self.df_dim*2, name='h1_conv'))
            input_h2 = lrelu(conv2d(input_h1, self.df_dim*4, name='h2_conv'))
            input_h3 = lrelu(conv2d(input_h2, self.df_dim*8, name='h3_conv'))
            print(input_h3.get_shape())
            input_h4 = lrelu(linear(tf.reshape(input_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            input_z = lrelu(linear(input_h4, featsize, 'hz_lin'))
            self.input_z = input_z
            input_zh0 = lrelu(linear(tf.concat([input_z, ctx_z], 1), featsize, 'zh0'))
            self.translated_z = linear(input_zh0, featsize, 'translate_z')
#             self.simloss = tf.zeros(())
#             print(self.input_z.get_shape())
#             self.simloss = 0
#             for j in range(3):
# #                 self.asdf = z[j*25:(j+1) * 25]- z[(j+1) * 25 : (j+2) * 25]
#                 self.simloss += tf.reduce_mean((input_z[j*25:(j+1) * 25]-
#                                                 input_z[(j+1) * 25 : (j+2) * 25]) ** 2)
#             mean, var = tf.nn.moments(input_z,axes=[0])
#             print(var.get_shape())
#             self.simloss /= tf.reduce_mean(var)
            scope.reuse_variables()

            truth_h0 = lrelu(conv2d(outputimg, self.df_dim, name='h0_conv'))
            truth_h1 = lrelu(conv2d(truth_h0, self.df_dim*2, name='h1_conv'))
            truth_h2 = lrelu(conv2d(truth_h1, self.df_dim*4, name='h2_conv'))
            truth_h3 = lrelu(conv2d(truth_h2, self.df_dim*8, name='h3_conv'))
            print(truth_h3.get_shape())
            truth_h4 = lrelu(linear(tf.reshape(truth_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            truth_z = lrelu(linear(truth_h4, featsize, 'hz_lin'))
            self.truth_z = truth_z
            self.simloss = tf.zeros(())
            print(self.truth_z.get_shape())

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            self.output_z_ = lrelu(linear(self.translated_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(self.output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(output_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            output_h2 = lrelu(deconv2d(output_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            output_h3 = lrelu(deconv2d(output_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(output_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            scope.reuse_variables()

            truthoutput_z_ = lrelu(linear(truth_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(truthoutput_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(truthoutput_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            truthoutput_h3 = lrelu(deconv2d(truthoutput_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            truthoutput_h4 = deconv2d(truthoutput_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

#         self.simloss = tf.nn.l2_loss(truthoutput_h2 - self.output_h2)
        self.simloss = tf.reduce_mean((self.translated_z - self.truth_z) ** 2) * 1e3
        mean, var = tf.nn.moments(self.truth_z,axes=[0])
        print(var.get_shape())
        self.simloss /= tf.reduce_mean(var)
        print(self.truth_z.get_shape())
        self.out = output_h4# + contextimg#tf.nn.tanh(h4)
        self.out2 = truthoutput_h4
        self.recon1 = tf.nn.l2_loss(outputimg - self.out)
        self.recon2 = tf.nn.l2_loss(outputimg - self.out2)
        self.loss = self.recon1 + self.recon2 + self.simloss
#         self.loss = self.reconloss# + self.simloss


class ContextAEFixed:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

        with tf.variable_scope("conv_input") as scope:
            input_h0 = lrelu(conv2d(inputimg, self.df_dim, name='h0_conv'))
            input_h1 = lrelu(conv2d(input_h0, self.df_dim*2, name='h1_conv'))
            input_h2 = lrelu(conv2d(input_h1, self.df_dim*4, name='h2_conv'))
            input_h3 = lrelu(conv2d(input_h2, self.df_dim*8, name='h3_conv'))
            print(input_h3.get_shape())
            input_h4 = lrelu(linear(tf.reshape(input_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            input_z = lrelu(linear(input_h4, featsize, 'hz_lin'))
            self.input_z = input_z

            with tf.variable_scope("trans") as scope2:
                ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
                ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
                ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
                ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
                ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
                ctx_z = linear(ctx_h4, featsize, 'hz_lin')
                input_zh0 = lrelu(linear(tf.concat([input_z, ctx_z], 1), featsize*2, 'zh0'))
                input_zh1 = lrelu(linear(input_zh0, featsize*2, 'zh1'))
                input_zh2 = lrelu(linear(input_zh1, featsize*2, 'zh2'))
                self.translated_z = linear(input_zh2, featsize, 'translate_z')
#             self.simloss = tf.zeros(())
#             print(self.input_z.get_shape())
#             self.simloss = 0
#             for j in range(3):
# #                 self.asdf = z[j*25:(j+1) * 25]- z[(j+1) * 25 : (j+2) * 25]
#                 self.simloss += tf.reduce_mean((input_z[j*25:(j+1) * 25]-
#                                                 input_z[(j+1) * 25 : (j+2) * 25]) ** 2)
#             mean, var = tf.nn.moments(input_z,axes=[0])
#             print(var.get_shape())
#             self.simloss /= tf.reduce_mean(var)
            scope.reuse_variables()

            truth_h0 = lrelu(conv2d(outputimg, self.df_dim, name='h0_conv'))
            truth_h1 = lrelu(conv2d(truth_h0, self.df_dim*2, name='h1_conv'))
            truth_h2 = lrelu(conv2d(truth_h1, self.df_dim*4, name='h2_conv'))
            truth_h3 = lrelu(conv2d(truth_h2, self.df_dim*8, name='h3_conv'))
            print(truth_h3.get_shape())
            truth_h4 = lrelu(linear(tf.reshape(truth_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            truth_z = lrelu(linear(truth_h4, featsize, 'hz_lin'))
            self.truth_z = truth_z
            self.simloss = tf.zeros(())
            print(self.truth_z.get_shape())

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            self.output_z_ = lrelu(linear(self.translated_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(self.output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(output_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            output_h2 = lrelu(deconv2d(output_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            output_h3 = lrelu(deconv2d(output_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(output_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            scope.reuse_variables()

            truthoutput_z_ = lrelu(linear(truth_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(truthoutput_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(truthoutput_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            truthoutput_h3 = lrelu(deconv2d(truthoutput_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            truthoutput_h4 = deconv2d(truthoutput_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

#         self.simloss = tf.nn.l2_loss(truthoutput_h2 - self.output_h2)
        self.simloss = tf.nn.l2_loss(self.translated_z - self.truth_z)
#         mean, var = tf.nn.moments(self.truth_z,axes=[0])
#         print(var.get_shape())
#         self.simloss /= tf.reduce_mean(var)
#         print(self.truth_z.get_shape())
        self.out = output_h4 #tf.nn.tanh(h4)
        self.out2 =  truthoutput_h4
        self.recon1 = tf.nn.l2_loss(outputimg - self.out)
        self.recon2 = tf.nn.l2_loss(outputimg - self.out2)
        self.loss = self.recon1 + self.recon2# + self.simloss
#         self.loss = self.reconloss# + self.simloss


class ContextSkipNew:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        srcimg = image[0]
        tgtimg = image[2]
        tgtctx = image[1]

        with tf.variable_scope("conv_context") as scope:
            tgtctx_h0 = lrelu(conv2d(tgtctx, self.df_dim, name='h0_conv'))
            tgtctx_h1 = lrelu(conv2d(tgtctx_h0, self.df_dim*2, name='h1_conv'))
            tgtctx_h2 = lrelu(conv2d(tgtctx_h1, self.df_dim*4, name='h2_conv'))
            tgtctx_h3 = lrelu(conv2d(tgtctx_h2, self.df_dim*8, name='h3_conv'))
            tgtctx_h4 = lrelu(linear(tf.reshape(tgtctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            tgtctx_z = linear(tgtctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv") as scope:
            srcimg_h0 = lrelu(conv2d(srcimg, self.df_dim, name='h0_conv'))
            srcimg_h1 = lrelu(conv2d(srcimg_h0, self.df_dim*2, name='h1_conv'))
            srcimg_h2 = lrelu(conv2d(srcimg_h1, self.df_dim*4, name='h2_conv'))
            srcimg_h3 = lrelu(conv2d(srcimg_h2, self.df_dim*8, name='h3_conv'))
            print(srcimg_h3.get_shape())
            srcimg_h4 = lrelu(linear(tf.reshape(srcimg_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            srcimg_z = lrelu(linear(srcimg_h4, featsize, 'hz_lin'))
            self.input_z = srcimg_z

            scope.reuse_variables()

            tgtimg_h0 = lrelu(conv2d(tgtimg, self.df_dim, name='h0_conv'))
            tgtimg_h1 = lrelu(conv2d(tgtimg_h0, self.df_dim*2, name='h1_conv'))
            tgtimg_h2 = lrelu(conv2d(tgtimg_h1, self.df_dim*4, name='h2_conv'))
            tgtimg_h3 = lrelu(conv2d(tgtimg_h2, self.df_dim*8, name='h3_conv'))
            tgtimg_h4 = lrelu(linear(tf.reshape(tgtimg_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            tgtimg_z = lrelu(linear(tgtimg_h4, featsize, 'hz_lin'))

        with tf.variable_scope("translate") as scope:
            trans_h0 = lrelu(linear(tf.concat([srcimg_z, tgtctx_z], 1), featsize, 'trans_h0'))
            trans_z = linear(trans_h0, featsize, 'trans_z')
            self.translated_z = trans_z

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            output_z_ = lrelu(linear(trans_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(tf.concat([output_h0, tgtctx_h3], 3),
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            output_h2 = lrelu(deconv2d(tf.concat([output_h1, tgtctx_h2], 3),
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            output_h3 = lrelu(deconv2d(tf.concat([output_h2, tgtctx_h1], 3),
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(tf.concat([output_h3, tgtctx_h0], 3),
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            scope.reuse_variables()

            truthoutput_z_ = lrelu(linear(tgtimg_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(tf.concat([truthoutput_h0, tgtctx_h3], 3),
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(tf.concat([truthoutput_h1, tgtctx_h2], 3),
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            truthoutput_h3 = lrelu(deconv2d(tf.concat([truthoutput_h2, tgtctx_h1], 3),
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            truthoutput_h4 = deconv2d(tf.concat([truthoutput_h3, tgtctx_h0], 3),
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

        self.simloss = tf.reduce_mean((trans_z - tgtimg_z) ** 2) * 1e3
        mean, var = tf.nn.moments(tgtimg_z, axes=[0])
        print(var.get_shape())
#         self.simloss /= tf.reduce_mean(var)
        print(tgtimg_z.get_shape())
        self.out = output_h4# + contextimg#tf.nn.tanh(h4)
        self.out2 = truthoutput_h4
        self.recon1 = tf.nn.l2_loss(tgtimg - self.out)
        self.recon2 = tf.nn.l2_loss(tgtimg - self.out2)
        self.loss = self.recon1 + self.recon2 + self.simloss

class ContextAETied:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

#         with tf.variable_scope("conv_context") as scope:
#             ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
#             ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
#             ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
#             ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
#             ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
#             ctx_z = linear(ctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv_input") as scope:
            input_h0 = lrelu(conv2d(inputimg, self.df_dim, name='h0_conv'))
            input_h1 = lrelu(conv2d(input_h0, self.df_dim*2, name='h1_conv'))
            input_h2 = lrelu(conv2d(input_h1, self.df_dim*4, name='h2_conv'))
            input_h3 = lrelu(conv2d(input_h2, self.df_dim*8, name='h3_conv'))
            print(input_h3.get_shape())
            input_h4 = lrelu(linear(tf.reshape(input_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            input_z = lrelu(linear(input_h4, featsize, 'hz_lin'))
            self.input_z = input_z

            scope.reuse_variables()

            ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
            ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
            ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
            ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
            ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            ctx_z = linear(ctx_h4, featsize, 'hz_lin')
#             self.simloss = tf.zeros(())
#             print(self.input_z.get_shape())
#             self.simloss = 0
#             for j in range(3):
# #                 self.asdf = z[j*25:(j+1) * 25]- z[(j+1) * 25 : (j+2) * 25]
#                 self.simloss += tf.reduce_mean((input_z[j*25:(j+1) * 25]-
#                                                 input_z[(j+1) * 25 : (j+2) * 25]) ** 2)
#             mean, var = tf.nn.moments(input_z,axes=[0])
#             print(var.get_shape())
#             self.simloss /= tf.reduce_mean(var)
            scope.reuse_variables()

            truth_h0 = lrelu(conv2d(outputimg, self.df_dim, name='h0_conv'))
            truth_h1 = lrelu(conv2d(truth_h0, self.df_dim*2, name='h1_conv'))
            truth_h2 = lrelu(conv2d(truth_h1, self.df_dim*4, name='h2_conv'))
            truth_h3 = lrelu(conv2d(truth_h2, self.df_dim*8, name='h3_conv'))
            print(truth_h3.get_shape())
            truth_h4 = lrelu(linear(tf.reshape(truth_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            truth_z = lrelu(linear(truth_h4, featsize, 'hz_lin'))
            self.truth_z = truth_z
            self.simloss = tf.zeros(())
            print(self.truth_z.get_shape())

        with tf.variable_scope("translate") as scope:
#             input_zh0 = lrelu(linear(tf.concat([input_z, ctx_z], 1), featsize, 'zh0'))
            self.translated_z = linear(tf.concat([input_z, ctx_z], 1), featsize, 'translate_z')

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            self.output_z_ = lrelu(linear(self.translated_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(self.output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(output_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            output_h2 = lrelu(deconv2d(output_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            output_h3 = lrelu(deconv2d(output_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(output_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            scope.reuse_variables()

            truthoutput_z_ = lrelu(linear(truth_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(truthoutput_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(truthoutput_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            truthoutput_h3 = lrelu(deconv2d(truthoutput_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            truthoutput_h4 = deconv2d(truthoutput_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

#         self.simloss = tf.nn.l2_loss(truthoutput_h2 - self.output_h2)
        self.simloss = tf.reduce_mean((self.translated_z - self.truth_z) ** 2) * 1e3
        mean, var = tf.nn.moments(self.truth_z,axes=[0])
        print(var.get_shape())
        self.simloss /= tf.reduce_mean(var)
        print(self.truth_z.get_shape())
        self.out = output_h4# + contextimg#tf.nn.tanh(h4)
        self.out2 = truthoutput_h4
        self.recon1 = tf.nn.l2_loss(outputimg - self.out)
        self.recon2 = tf.nn.l2_loss(outputimg - self.out2)
        self.loss = self.recon1 + self.recon2 + self.simloss
#         self.loss = self.reconloss# + self.simloss


keep_prob = 1.0

class ContextVanillaDrop:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        inputimg = image[0]
        contextimg = image[1]
        outputimg = image[2]

#         with tf.variable_scope("conv_context") as scope:
#             ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
#             ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
#             ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
#             ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
#             ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
#             ctx_z = linear(ctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv_input") as scope:
            input_h0 = lrelu(conv2d(inputimg, self.df_dim, name='h0_conv'))
            input_h1 = lrelu(conv2d(input_h0, self.df_dim*2, name='h1_conv'))
            input_h2 = lrelu(conv2d(input_h1, self.df_dim*4, name='h2_conv'))
            input_h3 = lrelu(conv2d(input_h2, self.df_dim*8, name='h3_conv'))
            print(input_h3.get_shape())
            input_h4 = lrelu(linear(tf.reshape(input_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            input_z = lrelu(linear(input_h4, featsize, 'hz_lin'))
            self.input_z = input_z

            scope.reuse_variables()

            ctx_h0 = lrelu(conv2d(contextimg, self.df_dim, name='h0_conv'))
            ctx_h1 = lrelu(conv2d(ctx_h0, self.df_dim*2, name='h1_conv'))
            ctx_h2 = lrelu(conv2d(ctx_h1, self.df_dim*4, name='h2_conv'))
            ctx_h3 = lrelu(conv2d(ctx_h2, self.df_dim*8, name='h3_conv'))
            ctx_h4 = lrelu(linear(tf.reshape(ctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            ctx_z = linear(ctx_h4, featsize, 'hz_lin')
#             self.simloss = tf.zeros(())
#             print(self.input_z.get_shape())
#             self.simloss = 0
#             for j in range(3):
# #                 self.asdf = z[j*25:(j+1) * 25]- z[(j+1) * 25 : (j+2) * 25]
#                 self.simloss += tf.reduce_mean((input_z[j*25:(j+1) * 25]-
#                                                 input_z[(j+1) * 25 : (j+2) * 25]) ** 2)
#             mean, var = tf.nn.moments(input_z,axes=[0])
#             print(var.get_shape())
#             self.simloss /= tf.reduce_mean(var)
            scope.reuse_variables()

            truth_h0 = lrelu(conv2d(outputimg, self.df_dim, name='h0_conv'))
            truth_h1 = lrelu(conv2d(truth_h0, self.df_dim*2, name='h1_conv'))
            truth_h2 = lrelu(conv2d(truth_h1, self.df_dim*4, name='h2_conv'))
            truth_h3 = lrelu(conv2d(truth_h2, self.df_dim*8, name='h3_conv'))
            print(truth_h3.get_shape())
            truth_h4 = lrelu(linear(tf.reshape(truth_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            truth_z = lrelu(linear(truth_h4, featsize, 'hz_lin'))
            self.truth_z = truth_z
            self.simloss = tf.zeros(())
            print(self.truth_z.get_shape())

        with tf.variable_scope("translate") as scope:
#             input_zh0 = lrelu(linear(tf.concat([input_z, ctx_z], 1), featsize, 'zh0'))
            self.translated_z = linear(tf.nn.dropout(tf.concat([input_z, ctx_z], 1), keep_prob), featsize, 'translate_z')

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            self.output_z_ = lrelu(linear(self.translated_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(self.output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(output_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            output_h2 = lrelu(deconv2d(output_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            output_h3 = lrelu(deconv2d(output_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(output_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            scope.reuse_variables()

            truthoutput_z_ = lrelu(linear(truth_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(truthoutput_h0,
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(truthoutput_h1,
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            truthoutput_h3 = lrelu(deconv2d(truthoutput_h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            truthoutput_h4 = deconv2d(truthoutput_h3,
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

#         self.simloss = tf.nn.l2_loss(truthoutput_h2 - self.output_h2)
        self.simloss = tf.reduce_mean((self.translated_z - self.truth_z) ** 2) * 1e3
        mean, var = tf.nn.moments(self.truth_z,axes=[0])
        self.var = var
        print(var.get_shape())
        print((var - np.ones(featsize,)).get_shape())
        self.regloss = tf.zeros(())#tf.reduce_mean((var - np.ones(featsize,)) ** 2) * 1e3
        print(self.truth_z.get_shape())
        self.out = output_h4# + contextimg#tf.nn.tanh(h4)
        self.out2 = truthoutput_h4
        self.recon1 = tf.nn.l2_loss(outputimg - self.out)
        self.recon2 = tf.nn.l2_loss(outputimg - self.out2)
        self.loss = self.recon1 + self.recon2 + self.simloss + self.regloss
#         self.loss = self.reconloss# + self.simloss

class ContextAEReal:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 100
        srcimg = image[0]
        tgtimg = image[2]
        tgtctx = image[1]

        nf0 = 32
        nf1 = 16
        nf2 = 16
        nf3 = 8
        ns0 = 1
        ns1 = 2
        ns2 = 1
        ns3 = 2
#         with tf.variable_scope("conv_context") as scope:

        def encode(img):
            img_h0 = lrelu(conv2d(img, nf0, d_h=ns0, d_w=ns0, name='h0_conv'))
            img_h1 = lrelu(conv2d(img_h0, nf1, d_h=ns1, d_w=ns1, name='h1_conv'))
            img_h2 = lrelu(conv2d(img_h1, nf2, d_h=ns2, d_w=ns2, name='h2_conv'))
            img_h3 = lrelu(conv2d(img_h2, nf3, d_h=ns3, d_w=ns3, name='h3_conv'))
            print(img_h3.get_shape())
            img_h4 = lrelu(linear(tf.nn.dropout(tf.reshape(img_h3, [self.batch_size, -1]), keep_prob), featsize, 'h4_lin'))
            img_z = lrelu(linear(tf.nn.dropout(img_h4, keep_prob), featsize, 'hz_lin'))
            return img_h0, img_h1, img_h2, img_h3, img_h4, img_z

        with tf.variable_scope("conv") as scope:
            srcimg_h0, srcimg_h1, srcimg_h2, srcimg_h3, srcimg_h4, srcimg_z = encode(srcimg)
            self.input_z = srcimg_z
            scope.reuse_variables()
            tgtimg_h0, tgtimg_h1, tgtimg_h2, tgtimg_h3, tgtimg_h4, tgtimg_z = encode(tgtimg)
            tgtctx_h0, tgtctx_h1, tgtctx_h2, tgtctx_h3, tgtctx_h4, tgtctx_z = encode(tgtctx)

        with tf.variable_scope("translate") as scope:
            trans_h0 = lrelu(linear(tf.nn.dropout(tf.concat([srcimg_z, tgtctx_z], 1), keep_prob), featsize, 'trans_h0'))
            trans_z = linear(tf.nn.dropout(trans_h0, keep_prob), featsize, 'trans_z')
            self.translated_z = trans_z

        s_h, s_w = self.output_height, self.output_width
        s_h0, s_h1, s_h2, s_h3 = \
            int(s_h/ns0), int(s_h/ns0/ns1), int(s_h/ns0/ns1/ns2), int(s_h/ns0/ns1/ns2/ns3)
        s_w0, s_w1, s_w2, s_w3 = \
            int(s_w/ns0), int(s_w/ns0/ns1), int(s_w/ns0/ns1/ns2), int(s_w/ns0/ns1/ns2/ns3)

        def decode(z, skip_h3, skip_h2, skip_h1, skip_h0):
            z_ = lrelu(linear(tf.nn.dropout(z, keep_prob), nf3*s_h3*s_w3, 'd_h0_lin'))
            h0 = tf.nn.dropout(tf.reshape(z_, [-1, s_h3, s_w3, nf3]), keep_prob)
            h1 = lrelu(deconv2d(tf.concat([h0, skip_h3], 3),
                    [self.batch_size, s_h2, s_w2, nf2], name='d_h1', d_h=ns3, d_w=ns3))
            h2 = lrelu(deconv2d(tf.concat([h1, skip_h2], 3),
                    [self.batch_size, s_h1, s_w1, nf1], name='d_h2', d_h=ns2, d_w=ns2))
            h3 = lrelu(deconv2d(tf.concat([h2, skip_h1], 3),
                    [self.batch_size, s_h0, s_w0, nf0], name='d_h3', d_h=ns1, d_w=ns1))
            print(h3.get_shape())
            h4 = deconv2d(tf.concat([h3, skip_h0], 3),
                    [self.batch_size, s_h, s_w, self.c_dim], name='d_h4', d_h=ns0, d_w=ns0)
            return h4
        with tf.variable_scope("deconv") as scope:
            output_h4 = decode(trans_z, tgtctx_h3, tgtctx_h2, tgtctx_h1, tgtctx_h0)
            scope.reuse_variables()
            truthoutput_h4 = decode(tgtimg_z, tgtctx_h3, tgtctx_h2, tgtctx_h1, tgtctx_h0)

        self.simloss = tf.reduce_mean((trans_z - tgtimg_z) ** 2) * 1e3
        print(tgtimg_z.get_shape())
        self.out = output_h4
        self.out2 = truthoutput_h4
        print(self.out.get_shape())
        self.recon1 = tf.nn.l2_loss(tgtimg - self.out)
        self.recon2 = tf.nn.l2_loss(tgtimg - self.out2)
        self.loss = self.recon1 + self.recon2 + self.simloss

class ContextAEInception:
    def __init__(self, gf_dim=256, df_dim=256,
                 gfc_dim=1024, dfc_dim=1024):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
#         self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width, self.c_dim = imgshape[-3:]
        self.batch_size = imgshape[1]
        featsize = 1024
        srcimg = image[0]
        tgtimg = image[2]
        tgtctx = image[1]
        
        with tf.variable_scope("conv_context") as scope:
            tgtctx_h0 = lrelu(conv2d(tgtctx, self.df_dim, name='h0_conv'))
            tgtctx_h1 = lrelu(conv2d(tgtctx_h0, self.df_dim*2, name='h1_conv'))
            tgtctx_h2 = lrelu(conv2d(tgtctx_h1, self.df_dim*4, name='h2_conv'))
            tgtctx_h3 = lrelu(conv2d(tgtctx_h2, self.df_dim*8, name='h3_conv'))
            tgtctx_h4 = lrelu(linear(tf.reshape(tgtctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            tgtctx_z = linear(tgtctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv") as scope:
            srcimg_h0 = lrelu(conv2d(srcimg, self.df_dim, name='h0_conv'))
            srcimg_h1 = lrelu(conv2d(srcimg_h0, self.df_dim*2, name='h1_conv'))
            srcimg_h2 = lrelu(conv2d(srcimg_h1, self.df_dim*4, name='h2_conv'))
            srcimg_h3 = lrelu(conv2d(srcimg_h2, self.df_dim*8, name='h3_conv'))
            print(srcimg_h3.get_shape())
            srcimg_h4 = lrelu(linear(tf.reshape(srcimg_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            srcimg_z = lrelu(linear(srcimg_h4, featsize, 'hz_lin'))
            self.input_z = srcimg_z
            
            scope.reuse_variables()
            
            tgtimg_h0 = lrelu(conv2d(tgtimg, self.df_dim, name='h0_conv'))
            tgtimg_h1 = lrelu(conv2d(tgtimg_h0, self.df_dim*2, name='h1_conv'))
            tgtimg_h2 = lrelu(conv2d(tgtimg_h1, self.df_dim*4, name='h2_conv'))
            tgtimg_h3 = lrelu(conv2d(tgtimg_h2, self.df_dim*8, name='h3_conv'))
            tgtimg_h4 = lrelu(linear(tf.reshape(tgtimg_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            tgtimg_z = lrelu(linear(tgtimg_h4, featsize, 'hz_lin'))

        with tf.variable_scope("translate") as scope:
            trans_h0 = lrelu(linear(tf.concat([srcimg_z, tgtctx_z], 1), featsize, 'trans_h0'))
            trans_z = linear(trans_h0, featsize, 'trans_z')
            self.translated_z = trans_z
        
        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
#             s_h2, s_h4, s_h8, s_h16 = \
#                 int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
#             s_w2, s_w4, s_w8, s_w16 = \
#                 int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)
            _, s_h2, s_w2, _ = tgtctx_h0.get_shape().as_list()
            _, s_h4, s_w4, _ = tgtctx_h1.get_shape().as_list()
            _, s_h8, s_w8, _ = tgtctx_h2.get_shape().as_list()
            _, s_h16, s_w16, _ = tgtctx_h3.get_shape().as_list()
    
            output_z_ = lrelu(linear(trans_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(tf.concat([output_h0, tgtctx_h3], 3),
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            output_h2 = lrelu(deconv2d(tf.concat([output_h1, tgtctx_h2], 3),
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            output_h3 = lrelu(deconv2d(tf.concat([output_h2, tgtctx_h1], 3),
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            output_h4 = deconv2d(tf.concat([output_h3, tgtctx_h0], 3),
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')
            
            scope.reuse_variables()
            
            truthoutput_z_ = lrelu(linear(tgtimg_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(tf.concat([truthoutput_h0, tgtctx_h3], 3),
                    [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(tf.concat([truthoutput_h1, tgtctx_h2], 3),
                [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
            truthoutput_h3 = lrelu(deconv2d(tf.concat([truthoutput_h2, tgtctx_h1], 3),
                [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
            truthoutput_h4 = deconv2d(tf.concat([truthoutput_h3, tgtctx_h0], 3),
                [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

        self.simloss = tf.reduce_mean((trans_z - tgtimg_z) ** 2) * 1e3
        mean, var = tf.nn.moments(tgtimg_z, axes=[0])
        print(var.get_shape())
#         self.simloss /= tf.reduce_mean(var)
        print(tgtimg_z.get_shape())
        self.out = output_h4 + tgtctx#tf.nn.tanh(h4)
        self.out2 = truthoutput_h4 + tgtctx
        self.recon1 = tf.nn.l2_loss(tgtimg - self.out)
        self.recon2 = tf.nn.l2_loss(tgtimg - self.out2)
        self.loss = self.recon1 + self.recon2 + self.simloss


class ContextAEInception2:
    def __init__(self, strides, kernels, filters):
        self.strides = strides
        self.kernels = kernels
        self.filters = filters

    def build(self, image):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width, self.c_dim = imgshape[-3:]
        self.batch_size = imgshape[1]
        featsize = 1024
        srcimg = image[0]
        tgtimg = image[2]
        tgtctx = image[1]
        s1, s2, s3, s4 = self.strides
        k1, k2, k3, k4 = self.kernels
        f1, f2, f3, f4 = self.filters
        
        def encode(img):
            img_h0 = lrelu(conv2d(img, f1, k1, k1, s1, s1, name='h0_conv'))
            img_h1 = lrelu(conv2d(img_h0, f2, k2, k2, s2, s2, name='h1_conv'))
            img_h2 = lrelu(conv2d(img_h1, f3, k3, k3, s3, s3, name='h2_conv'))
            img_h3 = lrelu(conv2d(img_h2, f4, k4, k4, s4, s4, name='h3_conv'))
            print(img_h3.get_shape())
            img_h4 = lrelu(linear(tf.reshape(img_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            img_z = lrelu(linear(img_h4, featsize, 'hz_lin'))
            return img_h0, img_h1, img_h2, img_h3, img_h4, img_z

        with tf.variable_scope("conv_context") as scope:
            tgtctx_h0, tgtctx_h1, tgtctx_h2, tgtctx_h3, tgtctx_h4, tgtctx_z = encode(tgtctx)

        with tf.variable_scope("conv") as scope:
            srcimg_h0, srcimg_h1, srcimg_h2, srcimg_h3, srcimg_h4, srcimg_z = encode(srcimg)
            self.input_z = srcimg_z
            scope.reuse_variables()
            tgtimg_h0, tgtimg_h1, tgtimg_h2, tgtimg_h3, tgtimg_h4, tgtimg_z = encode(tgtimg)

        with tf.variable_scope("translate") as scope:
            trans_h0 = lrelu(linear(tf.concat([srcimg_z, tgtctx_z], 1), featsize, 'trans_h0'))
            trans_z = linear(trans_h0, featsize, 'trans_z')
            self.translated_z = trans_z
        
        s_h, s_w = self.output_height, self.output_width
        _, s_h2, s_w2, _ = tgtctx_h0.get_shape().as_list()
        _, s_h4, s_w4, _ = tgtctx_h1.get_shape().as_list()
        _, s_h8, s_w8, _ = tgtctx_h2.get_shape().as_list()
        _, s_h16, s_w16, _ = tgtctx_h3.get_shape().as_list()

        def decode(z, skip_h3, skip_h2, skip_h1, skip_h0):
            z_ = lrelu(linear(z, f4*s_h16*s_w16, 'd_h0_lin'))
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, f4])
            h1 = lrelu(deconv2d(tf.concat([h0, skip_h3], 3),
                    [self.batch_size, s_h8, s_w8, f3], name='d_h1',
                    d_h=s4, d_w=s4, k_h=k4, k_w=k4))
            h2 = lrelu(deconv2d(tf.concat([h1, skip_h2], 3),
                    [self.batch_size, s_h4, s_w4, f2], name='d_h2',
                    d_h=s3, d_w=s3, k_h=k3, k_w=k3))
            h3 = lrelu(deconv2d(tf.concat([h2, skip_h1], 3),
                    [self.batch_size, s_h2, s_w2, f1], name='d_h3',
                    d_h=s2, d_w=s2, k_h=k2, k_w=k2))
            print(h3.get_shape())
            h4 = deconv2d(tf.concat([h3, skip_h0], 3),
                    [self.batch_size, s_h, s_w, self.c_dim], name='d_h4', 
                    d_h=s1, d_w=s1, k_h=k1, k_w=k1)
            return h4

        with tf.variable_scope("deconv") as scope:
            output_h4 = decode(trans_z, tgtctx_h3, tgtctx_h2, tgtctx_h1, tgtctx_h0)
            scope.reuse_variables()
            truthoutput_h4 = decode(tgtimg_z, tgtctx_h3, tgtctx_h2, tgtctx_h1, tgtctx_h0)


        # with tf.variable_scope("deconv") as scope:
    
        #     output_z_ = lrelu(linear(trans_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
        #     output_h0 = tf.reshape(output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        #     output_h1 = lrelu(deconv2d(tf.concat([output_h0, tgtctx_h3], 3),
        #             [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
        #     output_h2 = lrelu(deconv2d(tf.concat([output_h1, tgtctx_h2], 3),
        #         [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
        #     output_h3 = lrelu(deconv2d(tf.concat([output_h2, tgtctx_h1], 3),
        #         [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
        #     output_h4 = deconv2d(tf.concat([output_h3, tgtctx_h0], 3),
        #         [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')
            
        #     scope.reuse_variables()
            
        #     truthoutput_z_ = lrelu(linear(tgtimg_z, self.gf_dim*8*s_h16*s_w16, 'd_h0_lin'))
        #     truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        #     truthoutput_h1 = lrelu(deconv2d(tf.concat([truthoutput_h0, tgtctx_h3], 3),
        #             [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='d_h1'))
        #     truthoutput_h2 = lrelu(deconv2d(tf.concat([truthoutput_h1, tgtctx_h2], 3),
        #         [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='d_h2'))
        #     truthoutput_h3 = lrelu(deconv2d(tf.concat([truthoutput_h2, tgtctx_h1], 3),
        #         [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='d_h3'))
        #     truthoutput_h4 = deconv2d(tf.concat([truthoutput_h3, tgtctx_h0], 3),
        #         [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

        self.simloss = tf.reduce_mean((trans_z - tgtimg_z) ** 2) * 1e3
        mean, var = tf.nn.moments(tgtimg_z, axes=[0])
        print(var.get_shape())
#         self.simloss /= tf.reduce_mean(var)
        print(tgtimg_z.get_shape())
        self.out = output_h4 + tgtctx#tf.nn.tanh(h4)
        self.out2 = truthoutput_h4 + tgtctx
        self.recon1 = tf.nn.l2_loss(tgtimg - self.out)
        self.recon2 = tf.nn.l2_loss(tgtimg - self.out2)
        self.loss = self.recon1 + self.recon2 + self.simloss

autodc = None
tfinput = None
sess = None
goodmean = None
pcaW = None
tftime = None
def initialize():
    global autodc, tfinput, sess, goodmean, pcaW
    if autodc is not None:
        return
    batch_size = 25
    idims = (299, 299)
    image = tf.placeholder(tf.uint8, (batch_size, ) + idims + (3, ), name='image')
    image_trans = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_trans = tf.subtract(image_trans, 0.5)
    image_trans = tf.multiply(image_trans, 2.0)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        model = inception_v3.inception_v3(image_trans, num_classes=1001, is_training=False, dropout_keep_prob=1.0)
    variables_to_restore = slim.get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())
    restorer.restore(sess, "model/inception_v3.ckpt")
    bird = scipy.misc.imread('model/bird.jpg')
    bird = scipy.misc.imresize(bird, idims)
    logits = sess.run(model[0], {image:[bird]*batch_size})
    print(np.argsort(logits[0])[-20:])
    autodc = model
    tfinput = image

# def initialize():
#     global autodc, tfinput, sess, goodmean, pcaW
#     if autodc is not None:
#         return
#     idim = (36, 64)
#     batch_size=1
#     tfinput = tf.placeholder(tf.float32, (3, batch_size) + idim + (3, ), name='x')
#     autodc = ContextAEReal()
#     # with tf.variable_scope("valid") as scope:
#     autodc.build(tfinput)

#     config = tf.ConfigProto(
#         # device_count = {'GPU': 0}
#     )
#     config.gpu_options.allow_growth=True
#     sess = tf.Session(config=config)
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#     # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autodc.loss)
#     sess.run(tf.global_variables_initializer())
#     allloss = []
#     validloss = []
#     itr = 0
#     saver = tf.train.Saver()
#     saver.restore(sess, 'model/ctxskiprealnew62575')
    # saver.restore(sess, 'model/ctxgoal-1024-vanilla-vp-distract13397')

    # saver.restore(sess, 'model/ctxgoal-1024-vanilla-vp-distract62598')
    # saver.restore(sess, 'model/ctxgoal-1024-vanilla-vp52430')

    # saver.restore(sess, '/home/andrewliu/research/viewpoint/ctxgoal-1024-aefix33728')
    # saver.restore(sess, '/home/andrewliu/research/viewpoint/ctxstarrtgoal-1024-aefix109433')
    # goodmean = np.load('/home/andrewliu/research/viewpoint/ctxfeatsresgoal.npy')


# def initialize():
#     global autodc, tfinput, sess, goodmean, pcaW
#     if autodc is not None:
#         return
#     idim = (48, 48)
#     batch_size=1
#     tfinput = tf.placeholder(tf.float32, (3, batch_size) + idim + (3, ), name='x')
#     autodc = ContextAE()
#     # with tf.variable_scope("valid") as scope:
#     autodc.build(tfinput)

#     config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
#     config.gpu_options.allow_growth=True
#     sess = tf.Session(config=config)
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#     # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autodc.loss)
#     sess.run(tf.global_variables_initializer())
#     allloss = []
#     validloss = []
#     itr = 0
#     saver = tf.train.Saver()
#     saver.restore(sess, '/home/andrewliu/research/viewpoint/ctx1024-AE18833')
#     goodmean = np.load('/home/andrewliu/research/viewpoint/ctxfeatsmult.npy')

# def initialize():
#     global autodc, tfinput, sess, goodmean, pcaW
#     if autodc is not None:
#         return
#     idim = (64, 64)
#     batch_size=1
#     tfinput = tf.placeholder(tf.float32, (batch_size,) + idim + (3, ), name='x')
#     autodc = ReachAE()
#     # with tf.variable_scope("valid") as scope:
#     autodc.build(tfinput)

#     config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
#     config.gpu_options.allow_growth=True
#     sess = tf.Session(config=config)
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#     # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autodc.loss)
#     sess.run(tf.global_variables_initializer())
#     allloss = []
#     validloss = []
#     itr = 0
#     saver = tf.train.Saver()
#     saver.restore(sess, '/home/andrewliu/research/viewpoint/singleviewpoint1024-AE3412')
#     goodmean = np.load('/home/andrewliu/research/viewpoint/aefeatsconvsingle.npy')


# def initialize():
#     global autodc, tfinput, sess, goodmean, pcaW
#     if autodc is not None:
#         return
#     idim = (64, 64)
#     batch_size=1
#     tfinput = tf.placeholder(tf.float32, (batch_size,) + idim + (3, ), name='x')
#     autodc = AE()
#     autodc.build(tfinput)

#     config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
#     config.gpu_options.allow_growth=True
#     sess = tf.Session(config=config)
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#     # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autodc.loss)
#     sess.run(tf.global_variables_initializer())
#     allloss = []
#     validloss = []
#     itr = 0
#     saver = tf.train.Saver()
#     saver.restore(sess, '/home/andrewliu/research/viewpoint/singleviewpoint1024-AE9409')
#     goodmean = np.load('/home/andrewliu/research/viewpoint/aefeatssingle.npz')['arr_0']


# def initialize():
#     global autodc, tfinput, sess, goodmean, pcaW, tftime
#     if autodc is not None:
#         return
#     idim = (64, 64)
#     batch_size=1
#     tfinput = tf.placeholder(tf.float32, (batch_size, ) + idim + (3, ), name='x')
#     tftime = tf.placeholder(tf.float32, (batch_size, nclass), name='y')
#     autodc = TimeSoftmax()
#     autodc.build(tfinput, tftime)

#     config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
#     config.gpu_options.allow_growth=True
#     sess = tf.Session(config=config)
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#     optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autodc.loss)
#     sess.run(tf.global_variables_initializer())
#     allloss = []
#     validloss = []
#     itr = 0
#     saver = tf.train.Saver()
#     saver.restore(sess, '/home/andrewliu/research/viewpoint/singleviewpointtimesoftmax8084')

# def initialize():
#     global autodc, tfinput, sess, goodmean, pcaW, tftime
#     if autodc is not None:
#         return
#     idim = (64, 64)
#     batch_size=1
#     tfinput = tf.placeholder(tf.float32, (batch_size, ) + idim + (3, ), name='x')
#     tftime = tf.placeholder(tf.float32, (batch_size, 1), name='y')
#     autodc = TimePred()
#     # with tf.variable_scope("valid") as scope:
#     autodc.build(tfinput, tftime)

#     config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
#     config.gpu_options.allow_growth=True
#     sess = tf.Session(config=config)
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#     # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autodc.loss)
#     sess.run(tf.global_variables_initializer())
#     allloss = []
#     validloss = []
#     itr = 0
#     saver = tf.train.Saver()
#     saver.restore(sess, '/home/andrewliu/research/viewpoint/singleviewpointtime2328')


# def initialize():
#     global autodc, tfinput, sess, goodmean, pcaW
#     if autodc is not None:
#         return
#     idim = (64, 64)
#     batch_size=1
#     tfinput = tf.placeholder(tf.float32, (batch_size, ) + idim + (3, ), name='x')
#     tftime = tf.placeholder(tf.float32, (batch_size, 1), name='y')
#     autodc = TimeDC()
#     autodc.build(tfinput, tftime)

#     config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
#     config.gpu_options.allow_growth=True
#     sess = tf.Session(config=config)
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#     optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autodc.loss)
#     sess.run(tf.global_variables_initializer())
#     allloss = []
#     validloss = []
#     itr = 0
#     saver = tf.train.Saver()
#     saver.restore(sess, '/home/andrewliu/research/viewpoint/noviewpointtime7058')
#     # goodmean = np.load('./shaping/goodmean.npz')['arr_0']
#     # pcaW =  np.load('./shaping/goodw.npz')['arr_0']

# def initialize():
#     global autodc, tfinput, sess, goodmean, pcaW
#     if autodc is not None:
#         return
#     idim = (64, 64)
#     batch_size=1
#     tfinput = tf.placeholder(tf.float32, (2, batch_size) + idim + (3, ), name='x')
#     autodc = SubspaceAE()
#     autodc.build(tfinput)

#     config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
#     config.gpu_options.allow_growth=True
#     sess = tf.Session(config=config)
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#     # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autodc.loss)
#     sess.run(tf.global_variables_initializer())
#     allloss = []
#     validloss = []
#     itr = 0
#     saver = tf.train.Saver()
#     saver.restore(sess, '/home/andrewliu/research/viewpoint/twoviewpoint1024-AE3456')
#     goodmean = np.load('/home/andrewliu/research/viewpoint/subspacemean.npz')['arr_0']
#     # pcaW =  np.load('./shaping/goodw.npz')['arr_0']
