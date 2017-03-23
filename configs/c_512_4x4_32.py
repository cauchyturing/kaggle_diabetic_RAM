import numpy as np

from config import Config
from data import BALANCE_WEIGHTS
from layers import *

cnf = {
    'name': __name__.split('.')[-1],
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 32,
    'batch_size_test': 8,
    'balance_weights': np.array(BALANCE_WEIGHTS),
    'balance_ratio': 0.975,
    'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'weight_decay': 0.0000,
    'sigma': 0.25,
    'schedule': {
        0: 0.0001,
        100: 0.00001,
        #150: 0.00001,
        251: 'stop',
        #2: 'stop',
    },
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (4, 4),
    }
    args.update(kwargs)
    return conv_params(**args)

n = 32

layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    ] + ([(ToCC, {})] if CC else []) + [
    (BatchNormLayer, {}),
    (Conv2DLayer, cp(n, stride=(2, 2), partial_sum=4)),
    (BatchNormLayer, {}),
    (Conv2DLayer, cp(n, pad=2, partial_sum=3)),
    (BatchNormLayer, {}),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(2 * n, stride=(2, 2), partial_sum=4)),
    (BatchNormLayer, {}),
    (Conv2DLayer, cp(2 * n, pad=2, partial_sum=9)),
    (BatchNormLayer, {}),
    (Conv2DLayer, cp(2 * n, partial_sum=4)),
    (BatchNormLayer, {}),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(4 * n, pad=2, partial_sum=29)),
    (BatchNormLayer, {}),
    (Conv2DLayer, cp(4 * n, partial_sum=2)),
    (BatchNormLayer, {}),
    (Conv2DLayer, cp(4 * n, pad=2, partial_sum=29)),
    (BatchNormLayer, {}),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(8 * n, pad=2, partial_sum=5)),
    (BatchNormLayer, {}),
    (Conv2DLayer, cp(8 * n, partial_sum=7)),
    (BatchNormLayer, {}),
    (Conv2DLayer, cp(8 * n, pad=2, partial_sum=5)),
    (BatchNormLayer, {}),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(16 * n, partial_sum=3)),
    ] + ([(FromCC, {})] if CC else []) + [
    #(RMSPoolLayer, pool_params()),
    #(DropoutLayer, {'p': 0.5}),
    #(DenseLayer, dense_params(1024)),
    #(FeaturePoolLayer, {'pool_size': 2}),
    #(DropoutLayer, {'p': 0.5}),
    #(DenseLayer, dense_params(1024)),
    #(FeaturePoolLayer, {'pool_size': 2}),
    (GlobalPoolLayer, {}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
