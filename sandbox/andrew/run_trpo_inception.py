import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from rllab import config

stub(globals())

from distutils.dir_util import copy_tree
import numpy as np
import os, shutil
srcmodeldirs = ['../models/inceptionmodel/']
modeldir = 'model/'
if os.path.exists(modeldir):
    shutil.rmtree(modeldir)

for srcdir in srcmodeldirs:
    copy_tree(srcdir, modeldir)


# config.AWS_IMAGE_ID = "ami-7d23496b"#"ami-1263eb04"
# config.AWS_INSTANCE_TYPE = "g2.8xlarge"
# config.AWS_SPOT_PRICE = "2.6001"
# subnet = 'us-east-1d'

config.AWS_IMAGE_ID = "ami-20c1e740"
config.AWS_INSTANCE_TYPE = "g2.2xlarge"
config.AWS_SPOT_PRICE = "0.903"
subnet = 'us-west-1c'

# config.AWS_IMAGE_ID = "ami-ecdd408c"
# config.AWS_INSTANCE_TYPE = "g2.8xlarge"
# config.AWS_SPOT_PRICE = "2.601"
# subnet = 'us-west-2b'

# config.AWS_IMAGE_ID = "ami-b8f069d8"
# config.AWS_INSTANCE_TYPE = "g2.2xlarge"
# config.AWS_SPOT_PRICE = "0.601"
# subnet = 'us-west-2b'


config.AWS_NETWORK_INTERFACES = [
    dict(
        SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
        Groups=[config.ALL_SUBNET_INFO[subnet]["Groups"]],
        DeviceIndex=0,
        AssociatePublicIpAddress=True,
    )
]

def rand_real():
    vp = np.random.uniform(low=0, high=360)
    vangle = np.random.uniform(low=-40, high=-70)
    cam_dist = np.random.uniform(low=1.5, high=2.5)
    distlow = 0.4
    distobj = np.random.uniform(low=distlow, high=0.7)
    distmult = np.random.uniform(low=1.7, high=2.1)
    object_ = [-(distobj - distlow), 0.0]
    goal = [-(distobj * distmult - distlow - 0.5), 0.0]
    return dict(vp=vp, vangle=vangle, object=object_, goal=goal,
        cam_dist=cam_dist, imsize=(36, 64), name="real", modelname='model/ctxskiprealnew62575',
        meanfile='model/real_inception.npz', modeldata='model/vdata_realnew200.npy')

real_params = {
    "env" : "Pusher3DOFReal-v1",
    "rand" : rand_real,
}


def getcolor():
    color = np.random.uniform(low=0, high=1, size=3)
    while np.linalg.norm(color - np.array([1.,0.,0.])) < 0.5:
        color = np.random.uniform(low=0, high=1, size=3)
    return color

def rand_reach():
    vp = np.random.uniform(low=0, high=360)
    goal = np.concatenate([np.random.uniform(low=-1.1, high=-0.5, size=1),
                 np.random.uniform(low=0.5, high=1.1, size=1)]).tolist()
    armcolor = getcolor()
    bgcolor = getcolor()
    while np.linalg.norm(bgcolor - armcolor) < 0.5:
        bgcolor = np.random.uniform(low=0, high=1, size=3)
    armcolor = armcolor.tolist() + [1.0]
    bgcolor = bgcolor.tolist() + [1.0]
    geoms = []
    for i in range(5):
        pos_x = np.random.uniform(low=-0.9, high=0.9)
        pos_y = np.random.uniform(low=0, high=1.0)
        rgba = getcolor().tolist()
        isinv = 1 if np.random.random() > 0.5 else 0
        geoms.append([rgba+[isinv], pos_x, pos_y])
    return dict(vp=vp, bgcolor=bgcolor, armcolor=armcolor, goal=goal,
        imsize=(48, 48), geoms=geoms,
        name="reach", modelname='model/reachvpdistract130723',
        meanfile='model/reach_inception.npz', modeldata='model/reachdata_train.npy',
        experttheano='experttheano_reach.pkl')

reach_params = {
    "env" : "Reacher3DOF-v1",
    "rand" : rand_reach,
}

def rand_push():
    while True:
        object_ = [np.random.uniform(low=-1.0, high=-0.4),
                     np.random.uniform(low=0.3, high=1.2)]
        goal = [np.random.uniform(low=-1.2, high=-0.8),
                     np.random.uniform(low=0.8, high=1.2)]
        if np.linalg.norm(np.array(object_)-np.array(goal)) > 0.45: break
    geoms = []
    for i in range(5):
        pos_x = np.random.uniform(low=-0.9, high=0.9)
        pos_y = np.random.uniform(low=0, high=1.0)
        rgba = getcolor().tolist()
        isinv = 1 if np.random.random() > 0.5 else 0
        geoms.append([rgba+[isinv], pos_x, pos_y])
    vp = np.random.uniform(low=0, high=360)
    return dict(vp=vp, object=object_, goal=goal, imsize=(48, 48), geoms=geoms,
        name="push", modelname='model/ctxskipstartgoalvpdistract53723',
        meanfile='model/push_inception.npz', modeldata='model/greenctxstartgoalvpdistractvalid.npy',
        experttheano='experttheano_push.pkl')

push_params = {
    "env" : "Pusher3DOF-v1",
    "rand" : rand_push,
}

def rand_sweep():
    vp = np.random.uniform(low=0, high=360)
    return dict(vp=vp, imsize=(36, 64), name="sweep",
        experttheano='experttheano_clean.pkl',
        meanfile='model/cleaner_inception.npz')

sweep_params = {
    "env" : "Cleaner-v1",
    "rand" : rand_sweep,
}


def rand_strike():
    vp = np.random.uniform(low=0, high=360, size=10).tolist()
    angle = [45]#np.random.uniform(low=0, high=90, size=10).tolist()
    ball = np.array([0.5, -0.175])
    while True:
        goal = np.concatenate([
                np.random.uniform(low=0.15, high=0.7, size=1),
                np.random.uniform(low=0.1, high=1.0, size=1)])
        if np.linalg.norm(ball - goal) > 0.17:
            break
    return dict(vp=vp, goal=goal.tolist(), angle=angle,
        imsize=(64, 64), name="strike", nvp=1,
        modelname='model/model_90000_1408.57_1291.54_110.72',
        modeldata='model/vdata_train.npy',
        experttheano='expert_striker.pkl',
        meanfile='model/strike_inception.npz')

strike_params = {
    "env" : "Striker-v0",
    "rand" : rand_strike,
}

oracle_mode = dict(mode='oracle')
inceptionsame_mode = dict(mode='inceptionsame', imsize=(299, 299))
inception_mode = dict(mode='inception', imsize=(299, 299))
ours_mode = dict(mode='ours', scale=0.01)

seeds = [123]

for params in [strike_params]:
    for nvar in range(10):
        randparams = params['rand']()
        for modeparams in [inceptionsame_mode]:
            if modeparams == inceptionsame_mode and params == real_params:
                continue
            for layer in ['PreLogits', 'Mixed_7c']:#, 'Mixed_6c', 'Mixed_7c']:
                copyparams = randparams.copy()
                copyparams.update(modeparams)
                copyparams['layer'] = layer
                mdp = normalize(GymEnv(params['env'], **copyparams))
                for seed in seeds:
                    policy = GaussianMLPPolicy(
                        env_spec=mdp.spec,
                        hidden_sizes=(32, 32),
                        init_std=10
                    )

                    baseline = LinearFeatureBaseline(
                        mdp.spec,
                    )

                    batch_size = 50*250
                    algo = TRPO(
                        env=mdp,
                        policy=policy,
                        baseline=baseline,
                        batch_size=batch_size,
                        whole_paths=True,
                        max_path_length=50,
                        n_itr=200,
                        step_size=0.01,
                        subsample_factor=1.0,
                        **copyparams
                    )

                    run_experiment_lite(
                        algo.train(),
                        exp_prefix="r-inception-same-strike-std2",
                        n_parallel=4,
                        # dry=True,
                        snapshot_mode="all",
                        seed=seed,
                        mode="ec2_mujoco",
                        #terminate_machine=False
                    )
