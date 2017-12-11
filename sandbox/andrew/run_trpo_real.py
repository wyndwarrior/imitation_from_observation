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
srcmodeldirs = ['../ablation_data_paper/pushreal/']
modeldir = 'model/'
if os.path.exists(modeldir):
    shutil.rmtree(modeldir)

for srcdir in srcmodeldirs:
    copy_tree(srcdir, modeldir)


# config.AWS_IMAGE_ID = "ami-7d23496b"#"ami-1263eb04"
# config.AWS_INSTANCE_TYPE = "g2.8xlarge"
# config.AWS_SPOT_PRICE = "2.6001"
# subnet = 'us-east-1d'

config.AWS_IMAGE_ID = "ami-1e24027e"
config.AWS_INSTANCE_TYPE = "g2.2xlarge"
config.AWS_SPOT_PRICE = "0.7001"
subnet = 'us-west-1b'

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
        cam_dist=cam_dist, imsize=(36, 64), name="real",
        meanfile='model/real_inception_Mixed_7c.npz', modeldata='model/vdata_realnew200.npy')

real_params = {
    "env" : "Pusher3DOFReal-v1",
    "rand" : rand_real,
}

# oracle_mode = dict(mode='oracle')
# inception_mode = dict(mode='inception', imsize=(299, 299))
ours_mode = dict(mode='ours', mode2='ours', scale=0.01, modelname='model/pushreal_none/ablation_pushreal_None_30000')
ours_nofeat = dict(mode='ours', mode2='ours_nofeat', scale=0.01, ablation_type='nofeat', modelname='model/pushreal_none/ablation_pushreal_None_30000')
ours_noimage = dict(mode='ours', mode2='ours_noimage', scale=0.01, ablation_type='noimage', modelname='model/pushreal_none/ablation_pushreal_None_30000')
ab_l2 = dict(mode='ours', mode2='ab_l2', scale=0.01, modelname='model/pushreal_l2/ablation_pushreal_L2_30000')
ab_l2l3 = dict(mode='ours', mode2='ab_l2l3', scale=0.01, modelname='model/pushreal_l2l3/ablation_pushreal_L2L3_30000')
ab_l1 = dict(mode='ours', mode2='ab_l1', scale=0.01, modelname='model/pushreal_l1/ablation_pushreal_L1_30000')

seeds = [123]

for params in [real_params]:
    for nvar in range(10):
        randparams = params['rand']()
        for modeparams in [ab_l2]:#, ours_mode, ours_nofeat, ours_noimage, ab_l2l3, ab_l1]:
            copyparams = randparams.copy()
            copyparams.update(modeparams)
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
                    n_itr=100,
                    step_size=0.01,
                    subsample_factor=1.0,
                    **copyparams
                )

                run_experiment_lite(
                    algo.train(),
                    exp_prefix="r-real-ab3",
                    n_parallel=4,
                    # dry=True,
                    snapshot_mode="all",
                    seed=seed,
                    mode="ec2_mujoco",
                    #terminate_machine=False
                )
