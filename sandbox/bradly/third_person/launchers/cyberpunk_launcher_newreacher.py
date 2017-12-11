from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.bradly.third_person.policy.random_policy import RandomPolicy
from sandbox.bradly.third_person.algos.cyberpunk_trainer import CyberPunkTrainer
from sandbox.bradly.third_person.policy.expert_reacher import load_expert_reacher
from sandbox.bradly.third_person.envs.reacher import ReacherEnv #TODO: Make this randomize all the time
from sandbox.bradly.third_person.envs.reacher_two import ReacherTwoEnv #TODO: Make this randomize only once.

from sandbox.bradly.third_person.discriminators.discriminator import DomainConfusionVelocityDiscriminator
from sandbox.bradly.third_person.launchers.cyberpunk_launcher_newreacher_aws import AWSDummy
import joblib
import tensorflow as tf
from rllab.envs.gym_env import GymEnv

from rllab.misc.instrument import stub, run_experiment_lite

from rllab import config

stub(globals())

import numpy as np

config.AWS_IMAGE_ID = "ami-6df5d30d"
config.AWS_INSTANCE_TYPE = "g2.2xlarge"
config.AWS_SPOT_PRICE = "0.7001"
subnet = 'us-west-1c'

config.AWS_NETWORK_INTERFACES = [
        dict(
            SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
            Groups=[config.ALL_SUBNET_INFO[subnet]["Groups"]],
            DeviceIndex=0,
            AssociatePublicIpAddress=True,
            )
        ]

#
## novice_env = TfEnv(normalize(ReacherTwoEnv(), normalize_obs=True))
#expert_fail_pol = RandomPolicy(expert_env.spec)
#
#policy = GaussianMLPPolicy(
#    name="novice_policy",
#    env_spec=novice_env.spec,
#    # The neural network policy should have two hidden layers, each with 32 hidden units.
#    hidden_sizes=(32, 32)
#)
#
#baseline = LinearFeatureBaseline(env_spec=expert_env.spec)
#
#algo = TRPO(
#    env=novice_env,
#    policy=policy,
#    baseline=baseline,
#    batch_size=4000,
#    max_path_length=50,
#    n_itr=40,
#    discount=0.99,
#    step_size=0.01,
#    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
#
#)
#
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#with tf.Session(config=config) as sess:
#
#    #What do the n_itr and start_itr mean?
#    algo.n_itr = 0
#    algo.start_itr = 0
#    algo.train(sess=sess) #TODO: What is happening here?
#
#    im_height = 36
#    im_width = 64
#    im_channels = 3
#
#    dim_input = [im_height, im_width, im_channels]
#
#    disc = DomainConfusionVelocityDiscriminator(input_dim=dim_input, output_dim_class=2, output_dim_dom=2,
#                                                tf_sess=sess)
#
#    data = joblib.load("/home/andrewliu/research/viewpoint/rllab-tpil/third_person_im/data/local/experiment/experiment_2017_05_07_20_58_39_0001/itr_123.pkl")#"/home/abhigupta/abhishek_sandbox/viewpoint/third_person_im/data/local/experiment/experiment_2017_05_06_18_07_38_0001/itr_900.pkl")
#    expert_policy = data['policy']
#
#    # expert_policy = load_expert_reacher(expert_env, sess) #Load the expert #TODO: Need to train the expert
#
#    #from rllab.sampler.utils import rollout
#    #while True:
#    #        t = rollout(env=expert_env, agent=expert_policy, max_path_length=50, animated=True)
#
#    algo.n_itr = 40
#    trainer = CyberPunkTrainer(disc=disc, novice_policy_env=novice_env, expert_fail_pol=expert_fail_pol,
#                               expert_env=expert_env, novice_policy=policy,
#                               novice_policy_opt_algo=algo, expert_success_pol=expert_policy,
#                               im_width=im_width, im_height=im_height, im_channels=im_channels,
#                               tf_sess=sess, horizon=50)
#
#    iterations = 100
#    for iter_step in range(0, iterations):
#        trainer.take_iteration(n_trajs_cost=1000, n_trajs_policy=1000)
#
#    trainer.log_and_finish()
#
def getcolor():
    color = np.random.uniform(low=0, high=1, size=3)
    while np.linalg.norm(color - np.array([1.,0.,0.])) < 0.5:
        color = np.random.uniform(low=0, high=1, size=3)
    return color
for nvars in range(20):
# for mem in range(200, 500, 30):
    vp = np.random.uniform(low=0, high=360)
    goal = np.concatenate([np.random.uniform(low=-1.1, high=-0.5, size=1),
                 np.random.uniform(low=0.5, high=1.1, size=1)]).tolist()
    armcolor = getcolor()
    bgcolor = getcolor()
    while np.linalg.norm(bgcolor - armcolor) < 0.5:
        bgcolor = np.random.uniform(low=0, high=1, size=3)
    armcolor = armcolor.tolist() + [1.0]
    bgcolor = bgcolor.tolist() + [1.0]

    expert_env = TfEnv(GymEnv("Pusher3DOF-v1", force_reset=True, record_video=False))
    ## expert_env = TfEnv(normalize(ReacherEnv()))
    novice_env = TfEnv(GymEnv("Pusher3DOFNoChange-v1", force_reset=True, record_video=True,
        goal=goal, vp=vp, bgcolor=bgcolor, armcolor=armcolor))

    dummy = AWSDummy(expert_env=expert_env, novice_env=novice_env,
        horizon=50, itrs=100, trajs=100, expert_pkl='expert_reach.pkl')

    run_experiment_lite(dummy.run(),
            exp_prefix="cyberpunk_reach1",
            n_parallel=1,
            # dry=True,
            snapshot_mode="all",
            seed=1,
            mode="ec2_mujoco"
        )
