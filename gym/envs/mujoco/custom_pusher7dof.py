import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py.mjlib import mjlib

class CustomEnv7DOF(mujoco_env.MujocoEnv, utils.EzPickle):
    # def __init__(self, xml_path, image_width, image_height, speedup):
    def __init__(self, xml_path, viewer_params):
        utils.EzPickle.__init__(self)
        self._params = viewer_params
        mujoco_env.MujocoEnv.__init__(self, xml_path, viewer_params['speedup'])
        self._image_width = viewer_params['image_width']
        self._image_height = viewer_params['image_height']

    def _step(self, a):
        # vec_1 = self.get_body_com("object")-self.get_body_com("distal_4")
        vec_1 = self.get_body_com("object")-self.get_body_com("r_wrist_roll_link")
        vec_2 = self.get_body_com("object")-self.get_body_com("goal")
        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        #the coefficients in the following line are ad hoc
        reward = reward_dist + 0.1*reward_ctrl + 0.5*reward_near 
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid=0
        # self.viewer.cam.distance = 4.0
        cam_pos = self._params['cam_pos']
        self.viewer.cam.lookat[0] = cam_pos[0]
        self.viewer.cam.lookat[1] = cam_pos[1]
        self.viewer.cam.lookat[2] = cam_pos[2]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def _get_viewer(self):
        """Override mujoco_env method to put in the
        init_width and init_height

        """
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_width=self._image_width,
                init_height=self._image_height)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        while True:
            self.object = np.concatenate([self.np_random.uniform(low=-0.3, high=-0.05, size=1),
                                     self.np_random.uniform(low=0.25, high=0.65, size=1)])
            #self.goal = self.np_random.uniform(low=-1, high=1, size=2)
            self.goal = np.asarray([-0.05, 0.45])
            # if np.linalg.norm(self.object) > 0.7 and np.linalg.norm(self.goal) > 0.7:
            if np.linalg.norm(self.object-self.goal) > 0.20: break

        qpos[-4:-2] = self.object
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        #import IPython; IPython.embed()
        return self._get_obs()

    def _get_obs(self):
        # theta = self.model.data.qpos.flat[:-4]
        return np.concatenate([
            self.model.data.qpos.flat[:-4],
            self.model.data.qvel.flat[:-4],
            self.get_body_com("r_wrist_roll_link"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
        # return np.concatenate([
        #     np.sin(theta),
        #     np.cos(theta),
        #     self.model.data.qpos.flat[-4:],
        #     self.model.data.qvel.flat,
        #     self.get_body_com("object"),
        #     self.get_body_com("goal"),
        #     self.get_body_com("distal_4"),
        # ])
