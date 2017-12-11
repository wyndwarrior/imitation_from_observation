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

import numpy as np

config.AWS_IMAGE_ID = "ami-7d23496b"#"ami-1263eb04"
config.AWS_INSTANCE_TYPE = "g2.8xlarge"
config.AWS_SPOT_PRICE = "2.6001"
subnet = 'us-east-1d'

#config.AWS_IMAGE_ID = "ami-4382a523"
#config.AWS_INSTANCE_TYPE = "g2.2xlarge"
#config.AWS_SPOT_PRICE = "0.7001"
#subnet = 'us-west-1b'

# config.AWS_IMAGE_ID = "ami-ecdd408c"
# config.AWS_INSTANCE_TYPE = "g2.8xlarge"
# config.AWS_SPOT_PRICE = "2.601"
# subnet = 'us-west-2b'

# config.AWS_IMAGE_ID = "ami-b8f069d8"
# config.AWS_INSTANCE_TYPE = "g2.2xlarge"
# config.AWS_SPOT_PRICE = "0.601"
# subnet = 'us-west-2b'


# config.AWS_SECURITY_GROUPS = None
config.AWS_NETWORK_INTERFACES = [
    dict(
        SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
        Groups=[config.ALL_SUBNET_INFO[subnet]["Groups"]],
        DeviceIndex=0,
        AssociatePublicIpAddress=True,
    )
]

# Param ranges
seeds = [123]#[111,222,333,444,555]

configs = [
{"cam_dist": 2.343399506359302,
            "env_name": "Pusher3DOFReal-v1",
            "goal": [
              -0.2181554866957639,
              0.0
            ],
            "object": [
              -0.18427263878675026,
              0.0
            ],
            "scale": 0.01,
            "vangle": -52.94840634682497,
            "vp": 306.61260922055357},

    {"cam_dist": 2.413669421374702,
            "env_name": "Pusher3DOFReal-v1",
            "goal": [
              -0.24248563317988447,
              0.0
            ],
            "object": [
              -0.25756665879780527,
              0.0
            ],
            "scale": 0.01,
            "vangle": -54.128970888491956,
            "vp": 316.1840606644219},

            {"cam_dist": 2.1177303054204777,
            "env_name": "Pusher3DOFReal-v1",
            "goal": [
              -0.46625543263065705,
              0.0
            ],
            "object": [
              -0.2661080002347772,
              0.0
            ],
            "scale": 0.01,
            "vangle": -60.2010815701377,
            "vp": 98.81026264082921},


            {"cam_dist": 2.4661629855908687,
            "env_name": "Pusher3DOFReal-v1",
            "goal": [
              0.02263716546440686,
              0.0
            ],
            "object": [
              -0.01923199583962132,
              0.0
            ],
            "scale": 0.01,
            "vangle": -44.737606911893806,
            "vp": 262.82400533223927}]

# for config in configs:
#     cam_dist = config['cam_dist']
#     goal = config['goal']
#     object_ = config['object']
#     scale = config['scale']
#     vangle = config['vangle']
#     vp = config['vp']
    # if np.random.rand() < 1.0:
    #     vp = np.random.uniform(low=-30, high=30)
    # else:
    #     vp = np.random.uniform(low=180-30, high=180+30)

for nvar in range(10):
    # if np.random.rand() < 1.0:
    #     vp = np.random.uniform(low=-30, high=30)
    # else:
    #     vp = np.random.uniform(low=180-30, high=180+30)

    vp = np.random.uniform(low=0, high=360)
    vangle = np.random.uniform(low=-40, high=-70)
    cam_dist = np.random.uniform(low=1.5, high=2.5)
    distlow = 0.4
    distobj = np.random.uniform(low=distlow, high=0.7)
    distmult = np.random.uniform(low=1.7, high=2.1)
    object_ = [-(distobj - distlow), 0.0]
    goal = [-(distobj * distmult - distlow - 0.5), 0.0]

    for scale in [0.01]:#np.logspace(-4, 0, 5):
        mdp = normalize(GymEnv("Pusher3DOFReal-v1",
            vp=vp, vangle=vangle, object=object_, goal=goal, cam_dist=cam_dist, scale=scale))
        for seed in seeds:

            policy = GaussianMLPPolicy(
                env_spec=mdp.spec,
                hidden_sizes=(32, 32),#, 64),
                init_std=10
            )

            baseline = LinearFeatureBaseline(
                mdp.spec,
            )

            batch_size = 25000
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
            )

            run_experiment_lite(
                algo.train(),
                exp_prefix="realnew_inception2",
                n_parallel=6,
                # dry=True,
                snapshot_mode="all",
                seed=seed,
                # mode="ec2_mujoco",
                #terminate_machine=False
            )
