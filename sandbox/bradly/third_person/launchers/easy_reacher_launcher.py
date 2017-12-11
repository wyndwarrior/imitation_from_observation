from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.bradly.third_person.policy.random_policy import RandomPolicy
# from sandbox.bradly.third_person.algos.cyberpunk_trainer import CyberPunkTrainer
from sandbox.bradly.third_person.policy.expert_reacher import load_expert_reacher
from sandbox.bradly.third_person.envs.reacher_source import PusherEnv3DOF #TODO: Make this randomize all the time
from sandbox.bradly.third_person.envs.reacher_two import ReacherTwoEnv #TODO: Make this randomize only once. 
from rllab.sampler.utils import rollout
import joblib
# from sandbox.bradly.third_person.discriminators.discriminator import DomainConfusionVelocityDiscriminator
import numpy as np
import tensorflow as tf
from rllab.envs.gym_env import GymEnv


expert_env = TfEnv(GymEnv("Pusher3DOF-v1", force_reset=True, record_video=False))   
# data = joblib.load("itr_999.pkl")
# policy = data['policy']

# path = rollout(env, policy, max_path_length=args.max_length, animated=True, speedup=args.speedup)

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32),
    init_std=10
)

baseline = LinearFeatureBaseline(env_spec=expert_env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=500,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
algo.train()


import IPython
IPython.embed()