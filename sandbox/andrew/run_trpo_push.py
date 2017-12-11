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
srcmodeldirs = ['../models/modelskip/']
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

def getcolor():
    color = np.random.uniform(low=0, high=1, size=3)
    while np.linalg.norm(color - np.array([1.,0.,0.])) < 0.5:
        color = np.random.uniform(low=0, high=1, size=3)
    return color

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
    return dict(nvp=1, vp=vp, object=object_, goal=goal, imsize=(48, 48), geoms=geoms,
        name="push", modelname='model/ctxskipstartgoalvpdistract53723',
        meanfile=None, modeldata='model/greenctxstartgoalvpdistractvalid.npy')

push_params = {
    "env" : "Pusher3DOF-v1",
    "rand" : rand_push,
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
for params in [push_params]:
    for nvar in range(10):
        randparams = params['rand']()
        for modeparams in [ours_mode]:#, oracle_mode, gail_mode, tpil_mode, ]:
            for scale in [0.1, 1.0, 10.0]:#np.logspace(-2, 2, 5):
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
                            itrs=100,
                            trajs=250,
                            imsize=imsize,
                            expert_pkl='expert_push.pkl',
                            sanity=sanity,
                            **copyparams,
                        )
                    else:
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
                        exp_prefix="r_push_new_ours-quad1",
                        n_parallel=4,
                        # dry=True,
                        snapshot_mode="all",
                        seed=seed,
                        mode="ec2_mujoco",
                        # terminate_machine=False
                    )
