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
from sandbox.bradly.third_person.launchers.cyberpunk_aws_gail import CyberpunkAWSGAIL
from sandbox.bradly.third_person.launchers.cyberpunk_aws import CyberpunkAWS

stub(globals())

from distutils.dir_util import copy_tree
import numpy as np
import os, shutil


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


def rand_sweep():
    vp = np.random.uniform(low=0, high=360)
    return dict(vp=vp, imsize=(36, 64), name="sweep")

sweep_params = {
    "env" : "Cleaner-v1",
    "rand" : rand_sweep,
}

oracle_mode = dict(mode='oracle')
# inception_mode = dict(mode='inception', imsize=(299, 299))
ours_mode = dict(mode='ours', scale=1.0)
tpil_mode = dict(mode='tpil')

seeds = [123]

for params in [sweep_params]:
    for nvar in range(10):
        randparams = params['rand']()
        for modeparams in [tpil_mode]:
            for sanity in ['same', 'changing']:
                copyparams = randparams.copy()
                copyparams.update(modeparams)
                mdp = normalize(GymEnv(params['env'], **copyparams))
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
                        algo = CyberpunkAWSGAIL(
                            expert_env=mdp2,#normalize(GymEnv(params['env'], mode='tpil')),
                            novice_env=mdp,
                            horizon=50,
                            itrs=100,
                            trajs=250,
                            imsize=imsize,
                            expert_pkl='expert_sweep.pkl',
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
                            n_itr=100,
                            step_size=0.01,
                            subsample_factor=1.0,
                            **copyparams
                        )

                    run_experiment_lite(
                        algo.train(),
                        exp_prefix="r-sweep-gail3",
                        n_parallel=4,
                        # dry=True,
                        snapshot_mode="all",
                        seed=seed,
                        mode="ec2_mujoco",
                        #terminate_machine=False
                    )