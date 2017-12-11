from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import pickle
import os
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

filename = str(uuid.uuid4())

def all_videos(episode_number):
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Speedup')
    parser.add_argument('--loop', type=int, default=1,
                        help='# of loops')
    args = parser.parse_args()

    policy = None
    env = None
    while True:
        if ':' in args.file:
            # fetch file using ssh
            os.system("rsync -avrz %s /tmp/%s.pkl" % (args.file, filename))
            data = joblib.load("/tmp/%s.pkl" % filename)
            if policy:
                new_policy = data['policy']
                policy.set_param_values(new_policy.get_param_values())
                path = rollout(env, policy, max_path_length=args.max_length,
                               animated=True, speedup=args.speedup)
            else:
                policy = data['policy']
                env = data['env']
                path = rollout(env, policy, max_path_length=args.max_length,
                               animated=True, speedup=args.speedup)
        else:
            data = joblib.load(args.file)
            policy = data['policy']
            env = data['env']
            # import IPython
            # IPython.embed()
            


            # env._wrapped_env.env.monitor.close()
            # env._wrapped_env.env.monitor.start(env._wrapped_env._log_dir, all_videos, force=False, resume=True)
            # env._wrapped_env.env.monitor.configure(video_callable=all_videos)
            # # for r_no in range(args.loop):
            # while True:
            #     path = rollout(env, policy, max_path_length=args.max_length,animated=True, speedup=args.speedup)
            
            path = rollout(env, policy, max_path_length=args.max_length,animated=False, speedup=args.speedup)
            import IPython
            IPython.embed()
            import numpy as np
            print(np.sum(path['rewards']), np.sum(path['env_infos']['reward_true']))

            # break
            # import IPython
            # IPython.embed()
            # path = rollout(env, policy, max_path_length=args.max_length,animated=False, speedup=args.speedup)
            # import numpy as np
            # np.savez('goodtraj', traj=path['observations'])
        # print 'Total reward: ', sum(path["rewards"])
        args.loop -= 1
        if ':' not in args.file:
            if args.loop <= 0:
                break
