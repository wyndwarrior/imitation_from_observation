from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.bradly.third_person.envs.reacher import ReacherEnv
from rllab.envs.gym_env import GymEnv
stub(globals())
env = GymEnv("Reacher3DOF-v1", mode='oracle', force_reset=True)#, imsize=(48,48))   

# env = TfEnv(normalize(ReacherEnv()))
policy = GaussianMLPPolicy(
    # name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32),
    init_std=10
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=25000,
    max_path_length=50,
    n_itr=1000,
    discount=0.99,
    step_size=0.01,
    # imsize=(48,48),
    name='reach',
    mode='oracle',
    exp_prefix="reacher_state",
    # force_batch_sampler=True,
    # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

)
run_experiment_lite(
    algo.train(),
    n_parallel=6,
    seed=1,
)
