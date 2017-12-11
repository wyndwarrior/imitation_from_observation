from scripts.train_script import ModelTrainer

from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from rllab import config

stub(globals())

from distutils.dir_util import copy_tree
import numpy as np
import os, shutil
srcmodeldirs = ['../train/throw/']
modeldir = 'model/'
if os.path.exists(modeldir):
    shutil.rmtree(modeldir)
for srcdir in srcmodeldirs:
    copy_tree(srcdir, modeldir)

config.AWS_IMAGE_ID = "ami-5ce4944a"
config.AWS_INSTANCE_TYPE = "p2.xlarge"
config.AWS_SPOT_PRICE = "1.903"
subnet = 'us-east-1c'

trainer = ModelTrainer(idims=(64, 64), nvideos=7000,
    ntrain = 6000, batch_size=100, model='ContextSkipNew',
    nitr = 1000000, save_every = 10000)

run_experiment_lite(
    trainer.train(),
    exp_prefix="r-throw-train",
    # n_parallel=4,
    # dry=True,
    # snapshot_mode="all",
    # seed=seed,
    mode="ec2_mujoco",
    sync_s3_pkl=True,
    # terminate_machine=False
)