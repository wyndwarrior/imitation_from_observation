import numpy as np
from rllab.misc import tensor_utils
from rllab.sampler.base import BaseSampler
# from sandbox.rocky.analogy.utils import unwrap
from sandbox.rocky.tf.envs.base import TfEnv
from scipy.misc import imresize as imresize
import imageio
import rllab.misc.logger as logger
def unwrap(env):

    if isinstance(env, TfEnv):
        return unwrap(env.wrapped_env)
    return env

def savegif(name, frames):
    with imageio.get_writer(name, mode='I') as writer:
        for f in frames:
            writer.append_data(f.astype(np.uint8))


class CyberPunkTrainerGAIL:
    def __init__(self, disc, novice_policy_env, expert_env, novice_policy, novice_policy_opt_algo,
                 expert_success_pol, im_width, im_height, im_channels=3, tf_sess=None,
                 horizon=None):


        self.novice_policy_env = unwrap(novice_policy_env)
        self.expert_env = unwrap(expert_env)

        self.expert_success_pol = expert_success_pol
        self.novice_policy = novice_policy
        self.novice_policy_training_algo = novice_policy_opt_algo

        self.batch_size = 32
        self.horizon = horizon
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = im_channels
        self.iteration = 0

        self.disc = disc

        e_10 = np.zeros((2,))
        e_10[0] = 1
        self.expert_basis = e_10
        e_01 = np.zeros((2,))
        e_01[1] = 1
        self.novice_basis = e_01

        self.sampler = BaseSampler(self.novice_policy_training_algo)

        self.gan_rew_means = []
        self.true_rew_means = []

    def collect_trajs_for_cost(self, n_trajs, pol, env, cls):
        paths = []
        #print(n_trajs)
        for iter_step in range(0, n_trajs):
            paths.append(self.cyberpunk_rollout(agent=pol, env=env, max_path_length=self.horizon,
                                                reward_extractor=None))


        data_matrix = tensor_utils.stack_tensor_list([p['im_observations'] for p in paths])
        class_matrix = np.tile(cls, (n_trajs, self.horizon, 1))
        
        return dict(data=data_matrix, classes=class_matrix)

    def collect_trajs_for_policy(self, n_trajs, pol, env):
        paths = []
        for iter_step in range(0, n_trajs):
            paths.append(self.cyberpunk_rollout(agent=pol, env=env, max_path_length=self.horizon,
                                                reward_extractor=self.disc))
        return paths

    def take_iteration(self, n_trajs_cost, n_trajs_policy):
        expert_data = self.collect_trajs_for_cost(n_trajs=n_trajs_cost, pol=self.expert_success_pol,
                                                  env=self.expert_env, cls=self.expert_basis)
        on_policy_data = self.collect_trajs_for_cost(n_trajs=n_trajs_cost, pol=self.novice_policy,
                                                     env=self.novice_policy_env, cls=self.novice_basis)

        training_data_one, training_classes, training_time = self.shuffle_to_training_data(expert_data, on_policy_data)

        self.train_cost(training_data_one, training_classes, training_time, n_epochs=2)

        policy_training_paths = self.collect_trajs_for_policy(n_trajs_policy, pol=self.novice_policy, env=self.novice_policy_env)
        gan_rew_mean = np.mean(np.array([path['rewards'] for path in policy_training_paths]))
        gan_rew_std = np.std(np.array([path['rewards'] for path in policy_training_paths]))
        print('on policy GAN reward is ' + str(gan_rew_mean))
        true_rew_mean = np.mean(np.array([sum(path['true_rewards']) for path in policy_training_paths]))
        print('on policy True reward is ' + str(true_rew_mean))

        self.true_rew_means.append(true_rew_mean)
        self.gan_rew_means.append(gan_rew_mean)
        policy_training_samples = self.sampler.process_samples(itr=self.iteration, paths=policy_training_paths)
        self.novice_policy_training_algo.optimize_policy(itr=self.iteration, samples_data=policy_training_samples)

        self.iteration += 1
        print(self.iteration)

    def log_and_finish(self):
        print('true rews were ' + str(self.true_rew_means))
        print('gan rews were ' + str(self.gan_rew_means))

    def train_cost(self, data_one, classes, time, n_epochs):
        for iter_step in range(0, n_epochs):
            batch_losses = []
            for batch_step in range(0, data_one.shape[0], self.batch_size):
                data_batch = data_one[batch_step: batch_step+self.batch_size]
                classes_batch = classes[batch_step: batch_step+self.batch_size]
                time_batch = time[batch_step: batch_step+self.batch_size]
                batch_losses.append(self.disc.train([data_batch, time_batch], classes_batch))
            print('loss is ' + str(np.mean(np.array(batch_losses))))

    def shuffle_to_training_data(self, expert_data, on_policy_data):
        data = np.vstack([expert_data['data'], on_policy_data['data']])
        classes = np.vstack([expert_data['classes'], on_policy_data['classes']])

        sample_range = data.shape[0]*data.shape[1]
        all_idxs = np.random.permutation(sample_range)

        t_steps = data.shape[1]

        data_matrix = np.zeros(shape=(sample_range, self.im_height, self.im_width, self.im_channels))
        class_matrix = np.zeros(shape=(sample_range, 2))
        time_matrix = np.zeros(shape=(sample_range, 1))
        for one_idx, iter_step in zip(all_idxs, range(0, sample_range)):
            traj_key = int(np.floor(one_idx/t_steps))
            time_key = one_idx % t_steps
            data_matrix[iter_step, :, :, :] = data[traj_key, time_key, :, :, :]
            class_matrix[iter_step, :] = classes[traj_key, time_key, :]
            time_matrix[iter_step, 0] = time_key
        return data_matrix, class_matrix, time_matrix

    def cyberpunk_rollout(self, agent, env, max_path_length, reward_extractor=None, animated=False, speedup=1):
        height = self.im_height 
        width = self.im_width
        observations = []
        im_observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        #o = env.reset()
        o = env.reset()
        path_length = 0
        if animated:
            env.render()
        else:
            env.render(mode='rgb_array')

        while path_length < max_path_length:
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            observations.append(env.observation_space.flatten(o))
            rewards.append(r)
            actions.append(env.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o
            if animated:
                #TODO: Reshape here
                env.render()
                im = imresize(env.render(mode='rgb_array'), (height, width, 3))
                im_observations.append(im)
            else:
                im = imresize(env.render(mode='rgb_array'), (height, width, 3))
                im_observations.append(im)
                #timestep = 0.05
                #time.sleep(timestep / speedup)
        if animated:
            env.render(close=True)

        im_observations = tensor_utils.stack_tensor_list(im_observations)

        observations = tensor_utils.stack_tensor_list(observations)

        if reward_extractor is not None:
            true_rewards = tensor_utils.stack_tensor_list(rewards)
            # obs_pls_three = np.copy(im_observations)
            # for iter_step in range(0, obs_pls_three.shape[0]):  # cant figure out how to do this with indexing.
            #     idx_plus_three = min(iter_step+3, obs_pls_three.shape[0]-1)
            #     obs_pls_three[iter_step, :, :, :] = im_observations[idx_plus_three, :, :, :]
            # rewards = reward_extractor.get_reward(data=[im_observations, obs_pls_three], softmax=True)[:, 0]  # this is the prob of being an expert.
            #print(rewards)
            rewards = reward_extractor(data=[im_observations, np.linspace(0, im_observations.shape[0] -1, im_observations.shape[0])[:, None]], softmax=True)[:, 0]
        else:
            rewards = tensor_utils.stack_tensor_list(rewards)
            true_rewards = rewards

        return dict(
            observations=observations,
            im_observations=im_observations,
            actions=tensor_utils.stack_tensor_list(actions),
            rewards=rewards,
            true_rewards=true_rewards,
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )

