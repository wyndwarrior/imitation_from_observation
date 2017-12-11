from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.bradly.third_person.envs.reacher import ReacherEnv
from rllab.envs.gym_env import GymEnv
stub(globals())
env = TfEnv(GymEnv("Pusher3DOF-v1", mode='tpil', force_reset=True, record_video=False, imsize=(48,48)))   

# env = TfEnv(normalize(ReacherEnv()))
policy = GaussianMLPPolicy(
    name="policy",
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
    force_batch_sampler=True,
    # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

)
run_experiment_lite(
    algo.train(),
    n_parallel=4,
    seed=1,
)
