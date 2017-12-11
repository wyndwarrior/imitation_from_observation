import numpy as np
from gym import utils
import scipy.misc
from gym.envs.mujoco import mujoco_env

class StrikerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.2
        mujoco_env.MujocoEnv.__init__(self, 'striker.xml', 5)

    def _step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("r_wrist_flex_link")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        self._min_strike_dist = min(self._min_strike_dist, np.linalg.norm(vec_2))

        # print(np.linalg.norm(vec_1))
        if np.linalg.norm(vec_1) < self.strike_threshold and not self._striked:
            self._striked = True
            self._strike_pos = self.get_body_com("r_wrist_flex_link")
            self._strike_state = self.model.data.qpos.flat[:7]

        reward_penalty = 0
        if self._striked:
            vec_3 = self.get_body_com("r_wrist_flex_link") - self._strike_pos
            reward_near = - np.linalg.norm(vec_3)
            # reward_penalty = -np.linalg.norm(self._strike_state - 
            #     self.model.data.qpos.flat[:7])
            # print(reward_penalty)
            # reward_penalty = -np.linalg.norm(self.model.data.qvel.flat[:7])
        else:
            reward_near = - np.linalg.norm(vec_1)

        reward_dist = - np.linalg.norm(self._min_strike_dist)
        reward_ctrl = - np.square(a).sum()
        reward = 3 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near + reward_penalty

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        actual_dist = np.linalg.norm(vec_2)
        reward_true = 0
        if not hasattr(self, 'itr'):
            self.itr = 0
        if self.itr == 0:
            self.reward_orig = -actual_dist
        if self.itr == 49:
            reward_true = actual_dist/self.reward_orig

        imgs = None
        if self.itr % 2 == 1 and hasattr(self, "_kwargs") \
            and 'imsize' in self._kwargs and self._kwargs['mode'] != 'oracle':
            imgs = []
            nvp = self._kwargs['nvp']
            for vid in range(nvp):
                self.viewer_setup(vid=vid)
                img = self.render('rgb_array')
                idims = self._kwargs['imsize']
                img = scipy.misc.imresize(img, idims)
                # scipy.misc.imsave('test/_%d_%d.png'%(self.itr, vid), img)
                imgs.append(img)

        self.itr += 1
        return ob, 0, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl, reward_true=reward_true, imgs=imgs)

    def viewer_setup(self, vid=0):
        if self.viewer is None:
            self._get_viewer()
        self.viewer.cam.trackbodyid = 0
        rotation_angle = np.random.uniform(low=-0, high=360)
        viewing_angle = 45#np.random.uniform(low=0, high=90)
        if hasattr(self, "_kwargs") and 'vp' in self._kwargs:
            rotation_angle = self._kwargs['vp'][vid]
        if hasattr(self, "_kwargs") and 'angle' in self._kwargs:
            viewing_angle = self._kwargs['angle'][vid]
        cam_dist = 2.5
        cam_pos = np.array([0, 0.2, 0, cam_dist, -viewing_angle, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid=-1

    def reset_model(self):
        self.itr = 0
        self._min_strike_dist = np.inf
        self._striked = False
        self._strike_pos = None

        qpos = self.init_qpos

        self.ball = np.array([0.5, -0.175])
        while True:
            self.goal = np.array([0.7, 1.1])#np.concatenate([
                    # self.np_random.uniform(low=0.15, high=0.7, size=1),
                    # self.np_random.uniform(low=0.1, high=1.0, size=1)])
            if np.linalg.norm(self.ball - self.goal) > 0.17:
                break

        if hasattr(self, "_kwargs") and 'goal' in self._kwargs:
            self.goal = np.array(self._kwargs['goal'])

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1,
                size=self.model.nv)
        qvel[7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("r_wrist_flex_link"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
