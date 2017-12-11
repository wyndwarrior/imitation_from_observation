from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.bradly.third_person.policy.random_policy import RandomPolicy
from sandbox.bradly.third_person.algos.cyberpunk_trainer_gail import CyberPunkTrainerGAIL
from sandbox.bradly.third_person.discriminators.discriminator import ConvDiscriminator
import joblib
import tensorflow as tf
from rllab.envs.gym_env import GymEnv
import pickle
from rllab.misc.instrument import stub, run_experiment_lite
import rllab.misc.logger as logger
class CyberpunkAWSGAIL:

    def __init__(self, expert_env, novice_env, horizon, itrs, trajs, imsize, expert_pkl, **kwargs):
        self.expert_env = expert_env
        self.novice_env = novice_env
        self.horizon = horizon
        self.itrs = itrs
        self.trajs = trajs
        self.expert_pkl = expert_pkl
        self.imsize = imsize

    def train(self):

        expert_env = TfEnv(self.expert_env)#TfEnv(GymEnv("Pusher3DOF-v1", force_reset=True, record_video=False))
# expert_env = TfEnv(normalize(ReacherEnv()))
        novice_env = TfEnv(self.novice_env)#TfEnv(GymEnv("Pusher3DOFNoChange-v1", force_reset=True, record_video=True))

# novice_env = TfEnv(normalize(ReacherTwoEnv(), normalize_obs=True))
        expert_fail_pol = RandomPolicy(expert_env.spec)

        policy = GaussianMLPPolicy(
            name="novice_policy",
            env_spec=novice_env.spec,
            init_std=10,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(32, 32)
        )

        baseline = LinearFeatureBaseline(env_spec=expert_env.spec)

        algo = TRPO(
            env=novice_env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=self.horizon,
            n_itr=self.itrs,
            discount=0.99,
            step_size=0.01,
            optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:

            #What do the n_itr and start_itr mean?
            algo.n_itr = 0
            algo.start_itr = 0
            algo.train(sess=sess) #TODO: What is happening here?

            im_height = self.imsize[0]
            im_width = self.imsize[1]
            im_channels = 3

            dim_input = [im_height, im_width, im_channels]

            disc = ConvDiscriminator(input_dim=dim_input)

            #data = joblib.load(self.expert_pkl)#"/home/andrewliu/research/viewpoint/rllab-tpil/third_person_im/data/local/experiment/experiment_2017_05_07_20_58_39_0001/itr_123.pkl")#"/home/abhigupta/abhishek_sandbox/viewpoint/third_person_im/data/local/experiment/experiment_2017_05_06_18_07_38_0001/itr_900.pkl")
            #expert_policy = data['policy']
            with open(self.expert_pkl, 'rb') as pfile:
                expert_policy = pickle.load(pfile)
            # expert_policy = load_expert_reacher(expert_env, sess) #Load the expert #TODO: Need to train the expert

            #from rllab.sampler.utils import rollout
            #while True:
            #        t = rollout(env=expert_env, agent=expert_policy, max_path_length=50, animated=True)

            algo.n_itr = self.itrs
            trainer = CyberPunkTrainerGAIL(disc=disc, novice_policy_env=novice_env,
                                       expert_env=expert_env, novice_policy=policy,
                                       novice_policy_opt_algo=algo, expert_success_pol=expert_policy,
                                       im_width=im_width, im_height=im_height, im_channels=im_channels,
                                       tf_sess=sess, horizon=self.horizon)

            iterations = self.itrs
            for iter_step in range(0, iterations):
                logger.record_tabular('Iteration', iter_step)
                trainer.take_iteration(n_trajs_cost=self.trajs, n_trajs_policy=self.trajs)
                logger.dump_tabular(with_prefix=False)

            trainer.log_and_finish()

