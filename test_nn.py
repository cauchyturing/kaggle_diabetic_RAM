"""Conv Nets training script."""
import click
import numpy as np
np.random.seed(9)

import data
import util
from nn import create_net


#@click.command()
#@click.option('--cnf', default='configs/c_512_4x4_32.py', show_default=True,
#              help='Path or name of configuration module.')
#@click.option('--weights_from', default=None, show_default=True,
#              help='Path to initial weights file.')
def build(cnf, weights_from):

    config = util.load_module(cnf).config

    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)

    files = data.get_image_files(config.get('train_dir'))
    names = data.get_names(files)
    labels = data.get_labels(names).astype(np.float32)

    net = create_net(config)

    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")

    print("fitting ...")
   # net.fit(files, labels)
    return net, files, names, labels

cnf = 'configs/c_512_5x5_32.py'
cnf = 'configs1/c1_512_4x4_32.py'
weights_from = 'weights/c_512_5x5_32/best/0217_2017-02-28-17-51-48_0.302194565535.pkl'
weights_from = 'weights/c1_512_4x4_32/best/0219_2017-03-03-13-44-39_0.337270259857.pkl'
net, files, names, labels  = build(cnf=cnf, weights_from = weights_from)

