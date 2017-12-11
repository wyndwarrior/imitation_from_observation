import os
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
# os.environ["THEANO_FLAGS"] = "device=cpu"
# from rllab.envs.gym_env import GymEnv
# from rllab.envs.normalized_env import normalize

# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.envs.normalized_env import NormalizedEnv

# from rllab.algos.trpo import TRPO
# from rllab.misc.instrument import stub, run_experiment_lite
import itertools

from sandbox.rocky.tf.algos.im import Imitate
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.egreedyforward import EGreedyPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite

# stub(globals())

# Param ranges
seeds = range(1, 10)
# SwimmerGather hierarchical task
# mdp_classes = [SwimmerGatherEnv]
# mdps = [NormalizedEnv(env=mdp_class())
#         for mdp_class in mdp_classes]

env = TfEnv(normalize(GymEnv("Pusher3DOF-v1")))

for seed in seeds:

    # policy = GaussianMLPPolicy(
    #     env_spec=env.spec,
    #     hidden_sizes=(32, 32),#, 64),
    #     init_std=10
    # )

    # baseline = LinearFeatureBaseline(
    #     env.spec,
    # )

    # batch_size = 25000
    # algo = TRPO(
    #     env=mdp,
    #     policy=policy,
    #     baseline=baseline,
    #     batch_size=batch_size,
    #     whole_paths=True,
    #     max_path_length=50,
    #     n_itr=10000,
    #     step_size=0.01,
    #     subsample_factor=1.0,
    # )
    policy = EGreedyPolicy(
        env_spec=env.spec
    )
    algo = Imitate(
        env=env,
        policy=policy,
        baseline=None,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="forward_test",
        n_parallel=6,
        snapshot_mode="all",
        seed=seed,
        mode="local"
    )
