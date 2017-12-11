from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import pickle
import os
import numpy as np
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

filename = str(uuid.uuid4())

def all_videos(episode_number):
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument('logdir', type=str,
                        help='logdir')
    parser.add_argument('--max_length', type=int, default=50,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Speedup')
    parser.add_argument('--loop', type=int, default=1,
                        help='# of loops')
    args = parser.parse_args()

    with open(args.file, 'rb') as pfile:
        policy = pickle.load(pfile)
    while True:
        env = normalize(GymEnv(args.env))
        env._wrapped_env.env.monitor.start(args.logdir, all_videos, force=False, resume=True)
        env._wrapped_env.env.monitor.configure(video_callable=all_videos)
        path = rollout(env, policy, max_path_length=args.max_length, animated=False, speedup=args.speedup)
        vidpath = env._wrapped_env.env.monitor.video_recorder.path
        env._wrapped_env.env.monitor.close()
        if path is not None:
            true_rewards = np.sum(path['env_infos']['reward_true'])
            # if true_rewards < -0.2:
            #     os.remove(vidpath)
            print(true_rewards)

