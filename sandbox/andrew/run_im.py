import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from rllab.algos.cma_es import CMAES
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(1)
# SwimmerGather hierarchical task
# mdp_classes = [SwimmerGatherEnv]
# mdps = [NormalizedEnv(env=mdp_class())
#         for mdp_class in mdp_classes]

env = GymEnv("Pusher3DOF-v1")#, record_video=False)
mdp_classes = [env]
#mdp_classes = [SwimmerGatherEnv]
mdps = [normalize(mdp_class)
        for mdp_class in mdp_classes]

param_cart_product = itertools.product(
    mdps, seeds
)

for mdp, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32, 32),#, 64),
        min_std=1e-100,
        init_std=1e-100,
        learn_std=False
    )

    max_path_length=50
    batch_size = max_path_length * 500
    algo = CMAES(
        env=mdp,
        policy=policy,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=max_path_length,
        n_itr=1000
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="push_cmaes_ae",
        n_parallel=4,
        snapshot_mode="all",
        seed=seed,
        mode="local"
    )
