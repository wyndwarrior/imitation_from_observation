from scripts.train_script import ModelTrainer

from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from rllab import config

stub(globals())

from distutils.dir_util import copy_tree
import numpy as np
import os, shutil
srcmodeldirs = ['../train/strikebig/']
modeldir = 'model/'
if os.path.exists(modeldir):
    shutil.rmtree(modeldir)
for srcdir in srcmodeldirs:
    copy_tree(srcdir, modeldir)

config.AWS_IMAGE_ID = "ami-5ce4944a"
config.AWS_INSTANCE_TYPE = "p2.xlarge"
config.AWS_SPOT_PRICE = "1.903"
subnet = 'us-east-1c'

config.AWS_NETWORK_INTERFACES = [
    dict(
        SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
        Groups=[config.ALL_SUBNET_INFO[subnet]["Groups"]],
        DeviceIndex=0,
        AssociatePublicIpAddress=True,
    )
]

# trainer = ModelTrainer(idims=(299, 299), nvideos=100,
#     ntrain = 80, batch_size=25, model='ContextAEInception',
#     nitr = 1000, save_every = 600, nlen=25, nskip=2,
#     rescale=False, inception=True,
#     strides=[1,2,1,2], kernels=[3,3,3,3], filters=[1024, 1024, 512, 512])

trainer = ModelTrainer(idims=(299, 299), nvideos=2500,
    ntrain = 2300, batch_size=25, model='ContextAEInception',
    nitr = 100000, save_every = 5000, nlen=25, nskip=2,
    rescale=False, inception=True,
    strides=[1,2,1,2], kernels=[3,3,3,3], filters=[1024, 1024, 512, 512])

run_experiment_lite(
    trainer.train(),
    exp_prefix="r-strike-big-inception-train7c",
    # n_parallel=4,
    # dry=True,
    # snapshot_mode="all",
    # seed=seed,
    mode="ec2_mujoco",
    sync_s3_pkl=True,
    # terminate_machine=False
)