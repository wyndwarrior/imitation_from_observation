
from rllab.sampler.utils import rollout
import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import gym.envs.mujoco.arm_shaping
import scipy.misc
import pickle

class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        :param itr: Iteration number.
        :return: A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Terminate workers if necessary.
        """
        raise NotImplementedError

import tensorflow as tf
import gym.envs.mujoco.arm_shaping as arm_shaping

from nets import inception_v3
import scipy.misc
slim = tf.contrib.slim

class BaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self.initialized = False
        self.initialize()
    def initialize(self):
        if not hasattr(self.algo, '_kwargs'):
            return
        if 'ablation_type' in self.algo._kwargs:
            self.ablation_type = self.algo._kwargs['ablation_type']
        else:
            self.ablation_type = "None"
        #img = self.render('rgb_array')
        if 'imsize' in self.algo._kwargs:
            self.imsize = self.algo._kwargs['imsize']
        self.name = self.algo._kwargs['name']
        self.mode = self.algo._kwargs['mode']
        if self.mode.startswith('inception'):
            self.layer = self.algo._kwargs['layer']
            batch_size = 25
            idims = self.imsize
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
            self.sess = sess
            self.image = image
            self.model = model
            if self.mode == 'inceptionsame':
                with open(self.algo._kwargs['experttheano'], 'rb') as pfile:
                    expertpolicy = pickle.load(pfile)
                allfeats = []
                for nroll in range(20):
                    path = rollout(self.algo.env, expertpolicy, max_path_length=self.algo.max_path_length,animated=False)
                    # import IPython
                    # IPython.embed()
                    imgs = [img[0] for img in path['env_infos']['imgs'] if img is not None]
                    feat = self.sess.run(self.model[1][self.layer], {self.image: imgs})
                    allfeats.append(feat)
                # import IPython
                # IPython.embed()
                self.means = np.mean(allfeats, axis=0)
                self.std = np.std(allfeats, axis=0)
            else:
                data = np.load(self.algo._kwargs['meanfile'])
                self.means = data[self.layer]
                self.std = data[self.layer+'std']
                # print(efile, expertpolicy)
        elif self.mode.startswith('ours'):
            idim = self.imsize
            self.batch_size = 25
            tfinput = tf.placeholder(tf.uint8, (3, self.batch_size) + idim + (3, ), name='x')
            image_trans = tf.image.convert_image_dtype(tfinput, dtype=tf.float32)
            image_trans = tf.subtract(image_trans, 0.5)
            image_trans = tf.multiply(image_trans, 2.0)
            self.image_trans = image_trans
            if self.mode == 'oursinception':
                with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                    model = inception_v3.inception_v3(
                        tf.reshape(tensor=image_trans, shape=(3 *self.batch_size, ) + idim + (3, )),
                        num_classes=1001, is_training=False, dropout_keep_prob=1.0)
                autodc = arm_shaping.ContextAEInception2(strides=[1,2,1,2], kernels=[3,3,3,3], filters=[1024, 1024, 512, 512])
                featlayer = model[1]['Mixed_7c']
                featshape = featlayer.get_shape().as_list()
                featreshape = tf.reshape(featlayer, (3, self.batch_size, featshape[1], featshape[2], featshape[3]) )
                with tf.variable_scope("contextmodel") as scope:
                    autodc.build(featreshape)
                self.image_trans = featreshape
            else:
                if self.name == 'real' or self.name == 'sweep':
                    autodc = arm_shaping.ContextAEReal()
                elif self.name == 'push' or self.name == 'reach' or self.name == 'strike' or self.name == 'throw':
                    autodc = arm_shaping.ContextSkipNew()
                autodc.build(image_trans)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
            learning_rate = tf.placeholder(tf.float32, shape=[])
            # sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, self.algo._kwargs['modelname'])
            if self.mode == 'oursinception':
                bird = scipy.misc.imread('model/bird.jpg')
                bird = scipy.misc.imresize(bird, idim)
                logits = sess.run(model[0], {tfinput:[[bird]*self.batch_size]*3})
                print(np.argsort(logits[0])[-20:])
            self.nvp = self.algo._kwargs['nvp']
            self.sess = sess
            self.image = tfinput
            self.model = autodc

        self.initialized = True

    def __setstate__(self, d):
        print ("I'm being unpickled with these values:", d)
        self.__dict__ = d

    def __getstate__(self):
        # print ("I'm being pickled")
        return {}

    def process_samples(self, itr, paths):
        if not self.initialized:
            self.initialize()
        baselines = []
        returns = []

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            if hasattr(self.algo, '_kwargs'):
                if self.mode.startswith('inception'):
                    if idx % 100 == 0:
                        print("paths", idx)
                    imgs = [img[0] for img in path['env_infos']['imgs'] if img is not None]
                    feat = self.sess.run(self.model[1][self.layer], {self.image: imgs})
                    diff = self.means-feat
                    diff[self.std == 0] = 0
                    diff = diff ** 2 / (self.std + 1e-5)
                    means = np.mean(diff, axis=(1,2,3))
                    for j in range(25):
                        path["rewards"][j*2+1] -= means[j] * (j**2)
                elif self.mode == 'oracle':
                    path["rewards"] += path["env_infos"]["reward_true"]
                elif self.mode.startswith('ours'):
                    imgs = [img for img in path['env_infos']['imgs'] if img is not None]

                    if not hasattr(self, 'means'):
                        self.means = []
                        self.imgs = []
                        validdata = np.load(self.algo._kwargs['modeldata'])
                        for vp in range(self.nvp):
                            context = imgs[0][vp]
                            timgs = []
                            tfeats = []
                            nvideos = validdata.shape[1]
                            if self.mode == 'oursinception':
                                nvideos = 50
                            for i in range(nvideos):
                                if i % 10 == 0:
                                    print("feats", i)
                                skip = 1
                                if self.name == 'real' or self.name == 'sweep':
                                    skip = 2
                                if self.mode == 'oursinception':
                                    input_img = validdata[::skip, i]
                                else:
                                    input_img = ((validdata[::skip, i] + 1)*127.5).astype(np.uint8)
                                tfeat, timg = self.sess.run([self.model.translated_z, self.model.out],
                                    {self.image: [input_img, [context] * self.batch_size, 
                                        [context] * self.batch_size]})
                                timgs.append(timg)
                                tfeats.append(tfeat)
                            self.means.append(np.mean(tfeats, axis=0))
                            meanimgs = np.mean(timgs, axis=0)
                            self.imgs.append(meanimgs)
                            # for j in range(25):
                            #     scipy.misc.imsave('test/%d_%d.png' %(vp, j), arm_shaping.inverse_transform(meanimgs[j]))

                    if idx % 10 == 0:
                        print("feats", idx)
                    # import IPython
                    # IPython.embed()
                    costs = 0
                    for vp in range(self.nvp):
                        curimgs = [img[vp] for img in imgs]
                        feats, image_trans = self.sess.run([self.model.input_z, self.image_trans],
                            {self.image: [curimgs, [curimgs[0]] * self.batch_size, curimgs]})
                    # import IPython
                    # IPython.embed()
                    # for j in range(25):
                    #     scipy.misc.imsave('test/' + str(j) + "_recon.png", arm_shaping.inverse_transform(image_recon[j]))
                    # for j in range(25):
                    #     scipy.misc.imsave('test/' + str(j) + "_orig.png", arm_shaping.inverse_transform(image_trans[0][j]))

                        if self.ablation_type == "None":
                            costs += np.sum((self.means[vp] - feats)**2, axis = 1) + \
                                self.algo._kwargs['scale']*np.sum((self.imgs[vp] - image_trans[0])**2, axis = (1, 2, 3))
                        elif self.ablation_type == "nofeat":
                            costs = self.algo._kwargs['scale']*np.sum((self.imgs - image_trans[0])**2, axis = (1, 2, 3))
                        elif self.ablation_type == "noimage":
                            costs = np.sum((self.means - feats)**2, axis = 1)
                        elif self.ablation_type == 'recon':
                            costs = np.sum((self.means - feats)**2, axis = 1) + \
                                self.algo._kwargs['scale']*np.sum((image_recon - image_trans[0])**2, axis = (1, 2, 3))
                    # costs = np.sum((self.means - feats)**2, axis = 1) + \
                    #     self.algo._kwargs['scale']*np.sum((self.imgs - image_trans[0])**2, axis = (1, 2, 3))

                    for j in range(25):
                        path["rewards"][j*2+1] -= costs[j] * (j**2)


            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        if 'reward_true' in paths[0]['env_infos']:
            trues = [sum(path["env_infos"]["reward_true"]) for path in paths]
            logger.record_tabular('ReturnTrue',
                np.mean(trues))
            logger.record_tabular('MinTrue',
                np.min(trues))
            logger.record_tabular('MaxTrue',
                np.max(trues))
            logger.record_tabular('ArgmaxTrueReturn',
                trues[np.argmax(undiscounted_returns)])
        # logger.record_tabular('Shaping', np.mean([path["shaping_reward"] for path in paths]))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
