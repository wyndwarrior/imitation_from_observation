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


srcmodeldirs = ['../models/modelsweep/']
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


def rand_sweep():
    vp = np.random.uniform(low=0, high=360)
    return dict(vp=vp, imsize=(36, 64), name="sweep",
        modelname='model/ctxskipsweep29242', 
        modeldata='model/vdata_reducedsweep.npy')
    
sweep_params = {
    "env" : "Cleaner-v1",
    "rand" : rand_sweep,
}

ours_mode = dict(mode='ours', scale=1.0)
oracle_mode = dict(mode='oracle')

seeds = [123]

for params in [sweep_params]:
    for nvar in range(10):
        randparams = params['rand']()
        for modeparams in [oracle_mode, ours_mode]:
            for sanity in ['changing']:
                copyparams = randparams.copy()
                copyparams.update(modeparams)
                mdp = normalize(GymEnv(params['env'], **copyparams))
                if 'imsize' in copyparams:
                    imsize = copyparams['imsize']
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
                    exp_prefix="r-sweep1",
                    n_parallel=4,
                    # dry=True,
                    snapshot_mode="all",
                    seed=seed,
                    # mode="local",
                    mode="ec2_mujoco",
                    #terminate_machine=False
                )