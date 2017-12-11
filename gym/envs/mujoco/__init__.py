from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
#from gym.envs.mujoco.reacher_env2 import ReacherEnv2
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.pusher3dof import PusherEnv3DOF
from gym.envs.mujoco.reacher3dof import ReacherEnv3DOF
from gym.envs.mujoco.pusher3dofreal import PusherEnv3DOFReal
from gym.envs.mujoco.pusher7dof import PusherEnv7DOF
from gym.envs.mujoco.pusher7dof_exp import PusherEnv7DOFExp
from gym.envs.mujoco.pusher7dof_exp2 import PusherEnv7DOFExp2
from gym.envs.mujoco.textured_pusher7dof_exp2 import TexturedPusherEnv7DOFExp2
from gym.envs.mujoco.custom import CustomEnv
from gym.envs.mujoco.custom_pusher7dof import CustomEnv7DOF
from gym.envs.mujoco.custom_pusher7dof_exp import CustomEnv7DOFExp
from gym.envs.mujoco.custom_pusher7dof_exp2 import CustomEnv7DOFExp2
from gym.envs.mujoco.grasper import GrasperEnv
from gym.envs.mujoco.cleaner_env import CleanerEnv
#from gym.envs.mujoco.MultiViewPusher import MultiViewPusher

from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv