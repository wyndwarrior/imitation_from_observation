import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import scipy.misc
class ThrowerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self._ball_hit_ground = False
        self._ball_hit_location = None
        mujoco_env.MujocoEnv.__init__(self, 'thrower.xml', 5)

    def _step(self, a):
        ball_xy = self.get_body_com("ball")[:2]
        goal_xy = self.get_body_com("goal")[:2]

        if not self._ball_hit_ground and self.get_body_com("ball")[2] < -0.25:
            self._ball_hit_ground = True
            self._ball_hit_location = self.get_body_com("ball")

        if self._ball_hit_ground:
            ball_hit_xy = self._ball_hit_location[:2]
            reward_dist = -np.linalg.norm(ball_hit_xy - goal_xy)
        else:
            reward_dist = -np.linalg.norm(ball_xy - goal_xy)
        reward_ctrl = - np.square(a).sum()

        reward = reward_dist + 0.002 * reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        reward_true = 0
        reward_actual = -np.linalg.norm(ball_xy - goal_xy)
        if not hasattr(self, 'itr'):
            self.itr = 0
        if self.itr == 0:
            self.reward_orig = -reward_actual
        if self.itr == 49:
            reward_true = reward_actual/self.reward_orig


        img = None
        if self.itr % 2 == 1 and hasattr(self, "_kwargs") \
            and 'imsize' in self._kwargs and self._kwargs['mode'] != 'oracle':
            img = self.render('rgb_array')
            idims = self._kwargs['imsize']
            img = scipy.misc.imresize(img, idims)

        self.itr += 1

        return ob, 0, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl, reward_true=reward_true, img=img)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        rotation_angle = np.random.uniform(low=-0, high=360)
        if hasattr(self, "_kwargs") and 'vp' in self._kwargs:
            rotation_angle = self._kwargs['vp']
        cam_dist = 2.5
        cam_pos = np.array([0, 0.2, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid=-1

    def reset_model(self):
        self.itr = 0
        self._ball_hit_ground = False
        self._ball_hit_location = None

        qpos = self.init_qpos
        self.goal = np.array([self.np_random.uniform(low=-0.3, high=0.3),
                              self.np_random.uniform(low=-0.3, high=0.3)])
        if hasattr(self, "_kwargs") and 'goal' in self._kwargs:
            self.goal = np.array(self._kwargs['goal'])

        qpos[-9:-7] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("r_wrist_roll_link"),
            self.get_body_com("ball"),
            self.get_body_com("goal"),
        ])
