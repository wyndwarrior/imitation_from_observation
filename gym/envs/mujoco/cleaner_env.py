import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET
import os
import scipy.misc
# import gym.envs.mujoco.sweep_shaping

class CleanerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'cleaning_task.xml', 5, viewersize=(72*5, 128*5))

    def _step(self, a):  
        parm = self.get_body_com("distal_4")
        obj0 = self.get_body_com("object0")
        obj1 = self.get_body_com("object1")
        obj2 = self.get_body_com("object2")
        obj3 = self.get_body_com("object3")
        obj4 = self.get_body_com("object4")
        pgoal = self.get_body_com("goal")
        reward_obj0 = - np.linalg.norm(obj0-pgoal)
        reward_obj1 = - np.linalg.norm(obj1-pgoal)
        reward_obj2 = - np.linalg.norm(obj2-pgoal)
        reward_obj3 = - np.linalg.norm(obj3-pgoal)
        reward_obj4 = - np.linalg.norm(obj4-pgoal)

        reward_touch0 = - np.linalg.norm(parm-obj0)
        reward_touch1 = - np.linalg.norm(parm-obj1)
        reward_touch2 = - np.linalg.norm(parm-obj2)
        reward_touch3 = - np.linalg.norm(parm-obj3)
        reward_touch4 = - np.linalg.norm(parm-obj4)

        diff_xpos = -np.linalg.norm(self.model.data.site_xpos[0][1] - self.model.data.site_xpos[1][1])
        reward_ctrl = - np.square(a).sum()
        reward = reward_obj0 + reward_obj1 + reward_obj2 + reward_obj3 + reward_obj4 + 10*diff_xpos + \
                 reward_touch0 + reward_touch1 + reward_touch2 + reward_touch3 + reward_touch4 + \
                 0.001*reward_ctrl

        if not hasattr(self, 'itr'):
            self.itr = 0
        true_reward =  reward_obj0 + reward_obj1 + reward_obj2 + reward_obj3 + reward_obj4
        if self.itr == 0:
            self.reward_orig = -true_reward
        true_reward /= self.reward_orig
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        img = None
        if self.itr % 2 == 1 and hasattr(self, "_kwargs") \
            and 'imsize' in self._kwargs and self._kwargs['mode'] != 'tpil' and self._kwargs['mode'] != 'oracle':
            img = self.render('rgb_array')
            idims = self._kwargs['imsize']
            img = scipy.misc.imresize(img, idims)
        if self.itr != 49:
            true_reward = 0
        self.itr += 1
        return ob, 0, done, dict(reward_true=true_reward, img=img)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid=0
        self.viewer.cam.distance = 4.0
        rotation_angle = np.random.uniform(low=0, high=360)
        if hasattr(self, "_kwargs") and 'vp' in self._kwargs:
            rotation_angle = self._kwargs['vp']
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid=-1
        
    def reset_model(self):        
        self.itr = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])