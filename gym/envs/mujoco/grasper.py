import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class GrasperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, '3link_gripper_grasp.xml', 2)

    def _step(self, a):
        vec_1 = self.get_body_com("object")-self.get_body_com("distal_4")
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
        self.viewer.cam.trackbodyid=0
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.object = np.concatenate([self.np_random.uniform(low=-1.0, high=-0.4, size=1),
                self.np_random.uniform(low=0.5, high=1.2, size=1)])
            # self.object = np.concatenate([self.np_random.uniform(low=-1.0, high=1, size=1),
            #     self.np_random.uniform(low=0.2, high=1.0, size=1)])
            # self.goal = self.np_random.uniform(low=-1, high=1, size=2)
            # self.goal = np.asarray([0, 1.0])
            self.goal = np.asarray([-1.0, 0.8])
            if np.linalg.norm(self.object) > 0.7 and np.linalg.norm(self.goal) > 0.7:
                if np.linalg.norm(self.object-self.goal) > 0.5: break
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
            self.model.data.qpos.flat[-4:],
            self.model.data.qvel.flat[:-4],
            self.get_body_com("distal_4"),
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
