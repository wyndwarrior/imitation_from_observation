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

from sandbox.bradly.third_person.launchers.cyberpunk_aws import CyberpunkAWS
from sandbox.bradly.third_person.launchers.cyberpunk_aws_gail import CyberpunkAWSGAIL

stub(globals())

from distutils.dir_util import copy_tree
import numpy as np
import os, shutil
srcmodeldirs = ['../models/modelthrow/']
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
config.AWS_SPOT_PRICE = "0.703"
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

def rand_throw():
    vp = np.random.uniform(low=0, high=360)
    goal = np.array([np.random.uniform(low=-0.3, high=0.3),
                     np.random.uniform(low=-0.3, high=0.3)])
    return dict(vp=vp, imsize=(64, 64), name="throw", goal=goal.tolist(),
        modelname='model/model_70000_3007.74_2728.77_268.42', modeldata='model/vdata_train.npy')

throw_params = {
    "env" : "Thrower-v0",
    "rand" : rand_throw,
}

oracle_mode = dict(mode='oracle', mode2='oracle')
# inception_mode = dict(mode='inception', imsize=(299, 299))
ours_mode = dict(mode='ours', mode2='ours', scale=1.0)
ours_recon = dict(mode='ours', mode2='oursrecon', scale=1.0, ablation_type='recon')
tpil_mode = dict(mode='tpil', mode2='tpil')
gail_mode = dict(mode='tpil', mode2='gail')
ours_nofeat = dict(mode='ours', mode2='ours_nofeat', scale=1.0, ablation_type='nofeat')
ours_noimage = dict(mode='ours', mode2='ours_noimage', scale=1.0, ablation_type='noimage')

seeds = [123]

sanity = None
for params in [throw_params]:
    for nvar in range(10):
        randparams = params['rand']()
        for modeparams in [ours_mode]:
            for scale in [0.1, 1.0, 10.0]:
                copyparams = randparams.copy()
                copyparams.update(modeparams)
                copyparams['scale'] = scale
                mdp = normalize(GymEnv(params['env'], **copyparams))
                for seed in seeds:
                    policy = GaussianMLPPolicy(
                        env_spec=mdp.spec,
                        hidden_sizes=(32, 32),
                        init_std=1.0
                    )

                    baseline = LinearFeatureBaseline(
                        mdp.spec,
                    )

                    batch_size = 100*250
                    algo = TRPO(
                        env=mdp,
                        policy=policy,
                        baseline=baseline,
                        batch_size=batch_size,
                        whole_paths=True,
                        max_path_length=50,
                        n_itr=250,
                        step_size=0.01,
                        subsample_factor=1.0,
                        **copyparams
                    )

                run_experiment_lite(
                    algo.train(),
                    exp_prefix="r-throw-ours-scale",
                    n_parallel=4,
                    # dry=True,
                    snapshot_mode="all",
                    seed=seed,
                    mode="ec2_mujoco",
                    # terminate_machine=False
                )
