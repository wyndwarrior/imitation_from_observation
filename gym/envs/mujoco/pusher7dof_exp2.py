import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env 
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
# import gym.envs.mujoco.arm_shaping
import os
import IPython
import scipy.misc
class PusherEnv7DOFExp2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.randomize_xml('pr2_arm3d_blockpush_new_2.xml')
        mujoco_env.MujocoEnv.__init__(self, "temp.xml", 5)

    def _step(self, a):
        vec_1 = self.get_body_com("object")-self.get_body_com("r_wrist_roll_link")
        vec_2 = self.get_body_com("object")-self.get_body_com("goal")
        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        #the coefficients in the following line are ad hoc
        reward = reward_dist + 0.1*reward_ctrl# + 0.5*reward_near 
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        # if not hasattr(a, 'np_random'):
        #     self._seed()
        # self._get_viewer().render()
        # data, width, height = self._get_viewer().get_image()
        # img = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        # A = gym.envs.mujoco.arm_shaping
        # A.initialize()
        # input_img = A.transform(img)
        # # scipy.misc.imsave('test/' + str(self.itr) + "_.png", A.inverse_transform(input_img))
        # # import IPython
        # # IPython.embed()
        # # A.sess.run(A.autodc.out)
        # o = A.sess.run(A.autodc.z, {A.tfinput: [[input_img]]})
        # # scipy.misc.imsave('test/' + str(self.itr) + ".png", A.inverse_transform(o[0]))

        # self.itr += 1
        return ob, reward, done, dict(reward_true=reward_dist + 0.1*reward_ctrl)#, hid=o[0])

    def viewer_setup(self):
        # self.itr = 0
        # self.viewer.cam.trackbodyid=0
        # self.viewer.cam.distance = 4.0
        #Need to randomize
        # print("Set up")
        rotation_angle = self.np_random.uniform(low=0, high=0, size=1)
        cam_dist = self.np_random.uniform(low=3, high=3, size=1)
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid=-1


    def randomize_xml(self, xml_name):
        print("YOYO")
        fullpath = os.path.join(os.path.dirname(__file__), "assets", xml_name)
        newpath = os.path.join(os.path.dirname(__file__), "assets", "temp.xml")
        print(fullpath)
        tree = ET.parse(fullpath)
        root = tree.getroot()
        worldbody = tree.find(".//worldbody")
        num_objects = int(np.random.uniform(low=0, high=6, size=1))
        print("NUM objects %f"%(num_objects))
        for object_to_spawn in range(num_objects):

            pos_x = np.random.uniform(low=-0.9, high=0.9, size=1)
            pos_y = np.random.uniform(low=0, high=1.0, size=1)
            rgba_colors = np.random.uniform(low=0, high=1, size=3)
            ET.SubElement(
                worldbody, "geom",
                pos="%f %f -0.275"%(pos_x, pos_y),
                rgba="%f %f %f 1"%(rgba_colors[0], rgba_colors[1], rgba_colors[2]),
                name="object" + str(object_to_spawn),
                size="0.05 0.05 0.1",
                density='0.00001',
                type="cylinder",
                contype="0",
                conaffinity="0"

            )

        tree.write(newpath)


    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        while True:
            self.object = np.concatenate([self.np_random.uniform(low=-0.3, high=-0.05, size=1),
                                     self.np_random.uniform(low=0.25, high=0.65, size=1)])
            #self.goal = self.np_random.uniform(low=-1, high=1, size=2)
            self.goal = np.asarray([-0.05, 0.45])
            # if np.linalg.norm(self.object) > 0.7 and np.linalg.norm(self.goal) > 0.7:
            if np.linalg.norm(self.object-self.goal) > 0.17: break

        qpos[-4:-2] = self.object
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        #import IPython; IPython.embed()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:-4],
            self.model.data.qvel.flat[:-4],
            #self.get_body_com("r_wrist_roll_link"),
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
        # theta = self.model.data.qpos.flat[:-4]
        # return np.concatenate([
        #     np.sin(theta),
        #     np.cos(theta),
        #     self.model.data.qvel.flat[:-4],
        #     self.get_body_com("r_wrist_roll_link"),
        #     self.get_body_com("object"),
        #     self.get_body_com("goal"),
        # ])
