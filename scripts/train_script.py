import numpy as np
import tensorflow as tf
from tensorflow import gfile
import imageio
import pickle
import scipy.misc
import sys
import imageio
import os
import rllab.misc.logger as logger

slim = tf.contrib.slim
from nets import inception_v3
from gym.envs.mujoco import arm_shaping

def transform(image, resize_height, resize_width, rescale):
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    if rescale:
        return np.array(cropped_image)/127.5 - 1.
    return cropped_image
def inverse_transform(images):
    return (images+1.)/2.
def savegif(name, frames):
    with imageio.get_writer(name, mode='I') as writer:
        for f in frames:
            writer.append_data((np.clip(inverse_transform(f),0,1)*255).astype(np.uint8))

class ModelTrainer:
    def __init__(self, idims, nvideos, ntrain, batch_size, model, nitr,
        save_every, nlen, nskip, rescale, inception, strides, kernels, filters):
        self.idims = idims
        self.nvideos = nvideos
        self.ntrain = ntrain
        self.batch_size = batch_size
        self.nitr = nitr
        self.save_every = save_every
        self.nlen = nlen
        self.nskip = nskip
        self.rescale = rescale
        self.inception = inception
        self.strides = strides
        self.kernels = kernels
        self.filters = filters
        if model == 'ContextSkipNew':
            self.model = arm_shaping.ContextSkipNew
        elif model == 'ContextAEReal':
            self.model = arm_shaping.ContextAEReal
        elif model == 'ContextAEInception':
            self.model = arm_shaping.ContextAEInception2

    def train(self):
        logger.log(logger._snapshot_dir)
        basedir = logger._snapshot_dir
        if basedir is None:
            basedir = 'model/'
        else:
            basedir += "/"
        nlen = self.nlen
        videos = gfile.Glob("model/videos/*.mp4")
        # videos = pickle.load(open('videolist.pkl', 'rb'))
        idata = [[] for _ in range(nlen)]
        nfail = 0
        itr = 0
        np.random.shuffle(videos)
        for name in videos:
            try:
                vid = imageio.get_reader(name,  'ffmpeg')
                if itr % 100 == 0:
                    logger.log("%s %s" %(itr, len(idata[0])))
                if len(vid) == 51:
                    frames = []
                    for j in range(1, 51, self.nskip):
                        frame = transform(vid.get_data(j), self.idims[0], self.idims[1], self.rescale)
                        if not self.inception and np.max(frame) == -1:
                            logger.log("rip %s %s" % (itr, name))
                            frames = []
                            break
                        frames.append(frame)
                    if len(frames) != nlen:
                        continue
                    for j, f in enumerate(frames):
                        idata[j].append(f)
                else:
                    logger.log("%s %s" %(name, len(vid)))
                itr += 1
            except:
                nfail += 1
                logger.log("Unexpected error:" + str(sys.exc_info()))
                logger.log(name)
                if nfail > 10:
                    break
            if itr >= self.nvideos:
                break
        vdata = np.array(idata)
        np.save(basedir+'vdata_strike'+str(itr), vdata)
        logger.log(str(vdata.shape))

        # tf.reset_default_graph()
        batch_size = self.batch_size
        if self.inception:
            tfinput = tf.placeholder(tf.uint8, (3, batch_size, ) + self.idims + (3, ), name='image')
            image_trans = tf.image.convert_image_dtype(tf.reshape(tensor=tfinput, shape=(3 *batch_size, ) + self.idims + (3, )), dtype=tf.float32)
            image_trans = tf.subtract(image_trans, 0.5)
            image_trans = tf.multiply(image_trans, 2.0)
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                model = inception_v3.inception_v3(image_trans, num_classes=1001, is_training=False, dropout_keep_prob=1.0)
            variables_to_restore = slim.get_variables_to_restore()
            restorer = tf.train.Saver(variables_to_restore)
            bird = scipy.misc.imread('model/bird.jpg')
            bird = scipy.misc.imresize(bird, self.idims)
            test = self.model(strides=self.strides, kernels=self.kernels, filters=self.filters)
            featlayer = model[1]['Mixed_7c']
            featshape = featlayer.get_shape().as_list()
            featreshape = tf.reshape(featlayer, (3, batch_size, featshape[1], featshape[2], featshape[3]) )
            with tf.variable_scope("contextmodel") as scope:
                test.build(featreshape)
        else:
            tfinput = tf.placeholder(tf.float32, (3, batch_size) + self.idims + (3, ), name='x')
            test = self.model()
            with tf.variable_scope("contextmodel") as scope:
                test.build(tfinput)

        sess = tf.Session()
        learning_rate = tf.placeholder(tf.float32, shape=[])

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "contextmodel")
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(test.loss, var_list=train_vars)
        sess.run(tf.global_variables_initializer())
        allloss = []
        validloss = []
        itr = 0
        saver = tf.train.Saver()

        if self.inception:
            restorer.restore(sess, "model/inception_v3.ckpt")
            logits = sess.run(model[0], {tfinput:[[bird]*batch_size]*3})
            print(np.argsort(logits[0])[-20:])
            # vid = imageio.get_reader('/home/andrewliu/research/viewpoint/train/strikebig/videos/openaigym.video.1456.26585.video000000.mp4',  'ffmpeg')
            # frame = scipy.misc.imresize(vid.get_data(0), self.idims)
            # layersout = sess.run(model[1]['Mixed_5d'], {tfinput:[[frame]*batch_size]*3})
            # print(np.max(layersout))

        n = vdata.shape[1]
        ntrain = self.ntrain
        nvalid = n - ntrain
        logger.log("%s %s" %(ntrain, nvalid))
        nn_err = tf.reduce_sum(tf.abs(tf.argmin(tf.reduce_mean((featreshape[2][:, None]-test.out)**2, axis=(2,3,4)), axis=0) - np.arange(0, batch_size) % nlen))
        validdata = vdata[:, ntrain:]
        traindata = vdata[:, :ntrain]
        logger.log(str(validdata.shape) + str(traindata.shape))
        np.save(basedir+'vdata_train', traindata[:, :200])
        for itr in range(1, self.nitr):
            choicesrc = np.random.choice(ntrain, batch_size)
            choicetgt = np.random.choice(ntrain, batch_size)
            srcdata = traindata[np.arange(0, batch_size) % nlen, choicesrc]
            tgtdata = traindata[np.arange(0, batch_size) % nlen, choicetgt]
            tgtctx = traindata[0, choicetgt]
            batch = [srcdata, tgtctx, tgtdata]
            
        #     logger.log(sess.run( [test.recon1, test.recon2, test.loss, test.simloss], {tfinput: batch, learning_rate:1e-4, tftrain:False}))
            if itr % 4 == 0:
                _, loss, sim, r1, r2, err = sess.run( [optimizer, test.loss, test.simloss, test.recon1, test.recon2, nn_err], {tfinput: batch, learning_rate:1e-4})
                logger.log("%s %s %s %s %s %s" %(itr, loss, sim, r1, r2, err))
                allloss.append(loss)
            else:
                sess.run( optimizer, {tfinput: batch, learning_rate:1e-4})

            if itr % 40 == 0 or itr % self.save_every == 0:
                choicesrc = np.random.choice(nvalid, batch_size)
                choicetgt = np.random.choice(nvalid, batch_size)
                srcdata = validdata[np.arange(0, batch_size) % nlen, choicesrc]
                tgtdata = validdata[np.arange(0, batch_size) % nlen, choicetgt]
                tgtctx = validdata[0, choicetgt]
                batch = [srcdata, tgtctx, tgtdata]
                loss, sim, r1, r2, err = sess.run([test.loss, test.simloss, test.recon1, test.recon2, nn_err], {tfinput: batch})
                logger.log("%s %s %s %s %s %s E" %(itr, loss, sim, r1, r2, err))
                validloss.append(loss)
                if itr % self.save_every == 0:
                    os.mkdir(basedir+str(itr))
                    saver.save(sess, '%s%d/model_%d_%.2f_%.2f_%.2f_%d' %
                        (basedir, itr, itr, loss, r1, r2, err))
                    np.save('%s%d/validloss' % (basedir, itr), validloss)
                    if not self.inception:
                        for kk in range(10):
                            choicesrc = [np.random.randint(nvalid)] * batch_size
                            choicetgt = [np.random.randint(nvalid)] * batch_size
                            srcdata = validdata[np.arange(0, batch_size) % nlen, choicesrc]
                            tgtdata = validdata[np.arange(0, batch_size) % nlen, choicetgt]
                            tgtctx = validdata[0, choicetgt]
                            batch = [srcdata, tgtctx, tgtdata]
                            L, r1, r2, testouts = sess.run([test.loss, test.recon1, test.recon2, test.out], {tfinput: batch})
                            L, r1, r2, testouts2 = sess.run([test.loss, test.recon1, test.recon2, test.out2], {tfinput: batch})
                            savegif("%s%d/__%dtrans.gif"%(basedir, itr, kk), testouts[:nlen])
                            savegif("%s%d/__%drecon.gif"%(basedir, itr, kk), testouts2[:nlen])
                if itr >= self.save_every:
                    logger.record_tabular('Iteration', itr)
                    logger.record_tabular('Loss', loss)
                    logger.record_tabular('Sim', sim)
                    logger.record_tabular('R1', r1)
                    logger.record_tabular('R2', r2)
                    logger.record_tabular('NNErr', err)
                    logger.dump_tabular(with_prefix=False)
            # itr += 1
