import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET
import os

import scipy.misc
# import gym.envs.mujoco.arm_shaping
import scipy.misc
class PusherEnv3DOFReal(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, '3link_gripper_push_2d_real.xml', 5, viewersize=(72*5, 128*5))

    def _step(self, a):
        pobj = self.get_body_com("object")
        pgoal = self.get_body_com("goal")
        reward_dist = - np.linalg.norm(pgoal-pobj)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        if not hasattr(self, 'itr'):
            self.itr = 0

        if not hasattr(self, 'np_random'):
            self._seed()

        reward_true = 0
        if self.itr == 0:
            self.reward_orig = -reward_dist
        if self.itr == 49:
            reward_true = reward_dist/self.reward_orig

        img = None
        if self.itr % 2 == 1 and hasattr(self, "_kwargs") \
            and 'imsize' in self._kwargs and self._kwargs['mode'] != 'oracle':
            img = self.render('rgb_array')
            idims = self._kwargs['imsize']
            img = scipy.misc.imresize(img, idims)

        self.itr += 1
        return ob, 0, done, dict(reward_true=reward_true, img=img)

    def viewer_setup(self):
        # self.itr = 0
        self.viewer.cam.trackbodyid=0
        cam_dist = 3
        if hasattr(self, "_kwargs") and 'cam_dist' in self._kwargs:
            cam_dist = self._kwargs['cam_dist']
        self.viewer.cam.distance = cam_dist
        rotation_angle = 0
        if hasattr(self, "_kwargs") and 'vp' in self._kwargs:
            rotation_angle = self._kwargs['vp']
        view_angle = -45
        if hasattr(self, "_kwargs") and 'vangle' in self._kwargs:
            view_angle = self._kwargs['vangle']
        cam_pos = np.array([0, self.object[0], 0, cam_dist, view_angle, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid=-1

    def reset_model(self):
        self.itr = 0
        self.init_qpos[0] = 1.7
        self.init_qpos[1] = 3.0
        self.init_qpos[2] = 1.5
        qpos = self.init_qpos

        if hasattr(self, "_kwargs") and 'goal' in self._kwargs:
            self.object = np.array(self._kwargs['object'])
            self.goal = np.array(self._kwargs['goal'])
        else:
            self.object = np.array([0.0, 0.0])
            self.goal = np.array([0.0, 0.0])

        qpos[-4:-2] = self.object
        qpos[-2:] = self.goal
        qvel = self.init_qvel
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if not hasattr(self, 'np_random'):
            self._seed()
        if not hasattr(self, 'object'):
            self.reset_model()
        return np.concatenate([
            self.model.data.qpos.flat[:-4],
            self.model.data.qvel.flat[:-4],
            # self.get_body_com("distal_4"),
            # self.get_body_com("object"),
            # self.get_body_com("goal"),
        ])
