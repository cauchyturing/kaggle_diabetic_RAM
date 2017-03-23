from __future__ import division
import time

import click
import numpy as np

import nn
import data
import tta
import util

cnf='configs/c_512_5x5_32.py'
config = util.load_module(cnf).config
config.cnf['batch_size_train'] = 128

runs = {}
runs['train'] = config.get('train_dir')

net = nn.create_net(config)

weights_from = 'weights/c_512_5x5_32/weights_final.pkl'
net.load_params_from(weights_from)

tf, color_vecs = tta.build_quasirandom_transforms(1, skip=0, color_sigma=0.0, **data.no_augmentation_params)
for i, (tf, color_vec) in enumerate(zip(tfs, color_vecs), start=1):
    pass



