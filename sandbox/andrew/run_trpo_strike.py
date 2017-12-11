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
srcmodeldirs = ['../models/strikeinc/']
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
        modeldata='model/vdata_train.npy')

strike_params = {
    "env" : "Striker-v0",
    "rand" : rand_strike,
}

oracle_mode = dict(mode='oracle', mode2='oracle')
# inception_mode = dict(mode='inception', imsize=(299, 299))
oursinception_mode = dict(mode='oursinception', mode2='oursinception', scale=0.1, imsize=(299, 299),
    modelname='model/model_70000_225002.77_128751.15_96043.16_0')
ours_mode = dict(mode='ours', mode2='ours', scale=0.1)
ours_recon = dict(mode='ours', mode2='oursrecon', scale=1.0, ablation_type='recon')
tpil_mode = dict(mode='tpil', mode2='tpil', imsize=(48, 48))
gail_mode = dict(mode='tpil', mode2='gail')
ours_nofeat = dict(mode='ours', mode2='ours_nofeat', scale=1.0, ablation_type='nofeat')
ours_noimage = dict(mode='ours', mode2='ours_noimage', scale=1.0, ablation_type='noimage')

seeds = [123]

sanity = 'changing'
for params in [strike_params]:
    for nvar in range(5):
        randparams = params['rand']()
        for modeparams in [oursinception_mode]:
            for scale in [0.0, 0.1, 100.0]:#[1.0, 10.0, 100.0, 0.1]:
                copyparams = randparams.copy()
                copyparams.update(modeparams)
                copyparams['scale'] = scale
                mdp = normalize(GymEnv(params['env'], **copyparams))
                if copyparams['mode'] == 'tpil':
                    if sanity == 'change1':
                        copyparams = params['rand']()
                        copyparams.update(modeparams)
                        mdp2 = normalize(GymEnv(params['env'], **copyparams))
                    elif sanity == 'same':
                        mdp2 = mdp
                    elif sanity == 'changing':
                        mdp2 = normalize(GymEnv(params['env'], mode='tpil'))
                if 'imsize' in copyparams:
                    imsize = copyparams['imsize']
                for seed in seeds:
                    if copyparams['mode'] == 'tpil':
                        del copyparams['imsize']
                        awsalgo = CyberpunkAWS
                        if modeparams == gail_mode:
                            awsalgo = CyberpunkAWSGAIL
                        algo = awsalgo(
                            expert_env=mdp2,#normalize(GymEnv(params['env'], mode='tpil')),
                            novice_env=mdp,
                            horizon=50,
                            itrs=200,
                            trajs=250,
                            imsize=imsize,
                            expert_pkl='expert_striker.pkl',
                            sanity=sanity,
                            **copyparams,
                        )
                    else:
                        policy = GaussianMLPPolicy(
                            env_spec=mdp.spec,
                            hidden_sizes=(32, 32),
                            init_std=1.0
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
                    exp_prefix="r-strike-ours-inception-7c-quad2",
                    n_parallel=4,
                    # dry=True,
                    snapshot_mode="all",
                    seed=seed,
                    mode="ec2_mujoco",
                    # terminate_machine=False
                )
