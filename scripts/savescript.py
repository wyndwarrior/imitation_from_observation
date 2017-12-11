from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import pickle
import os
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import tensorflow as tf
filename = str(uuid.uuid4())

def all_videos(episode_number):
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('loadfile', type=str,
                        help='path to the snapshot file')
    parser.add_argument('savefile', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    tf.InteractiveSession()
    data = joblib.load(args.loadfile)
    with open(args.savefile + '.pkl', 'wb') as pfile:
        pickle.dump(data['policy'], pfile)
