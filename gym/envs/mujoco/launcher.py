import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env 

class LauncherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'launcher.xml', 5)

    def _step(self, a):
        vec_1 = self.get_body_com("ball")-self.get_body_com("goal")
        reward_dist = - np.linalg.norm(vec_1)
        reward_ctrl = - np.square(a).sum()

        #the coefficients in the following line are ad hoc
        reward = reward_dist + 0.1*reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid=0
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # self.goal = np.asarray([-0.05, 0.45])
        self.goal = np.asarray([0.1, 0.75])

        qpos[-8:-6] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-8:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:-8],
            self.model.data.qvel.flat[:-8],
            self.get_body_com("goal"),
        ])
