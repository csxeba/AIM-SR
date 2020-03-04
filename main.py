"""
This module contains the logic for instantiating and fitting an CNN.
"""

import numpy as np

from brainforge import Backpropagation, LayerStack
from brainforge import layers
from brainforge import optimizers

import streamer

# Low batch size with relatively high LR will act as a regularizer.
BATCH_SIZE = 10
LEARNING_RATE = 1e-3

# Brainforge convolution expects channels first format.
stream = streamer.Stream(image_format="channels_first")

# I chose a fully convolutional architecture for this classification task.
# Due to the low number of parameters of the Conv layer
# omitting the Dense layers will act as a regularizer.
# Note: the network expects BGR inputs.

stack = LayerStack(stream.input_shape, layers=[
    
    layers.ConvLayer(nfilters=16, filterx=5, filtery=5, compiled=True),
    layers.PoolLayer(filter_size=2, compiled=True),
    layers.Activation("relu"),

    layers.ConvLayer(nfilters=32, filterx=5, filtery=5, compiled=True),
    layers.Activation("relu"),

    layers.ConvLayer(nfilters=32, filterx=5, filtery=5, compiled=True),
    layers.PoolLayer(filter_size=2, compiled=True),
    layers.Activation("relu"),

    layers.ConvLayer(nfilters=stream.NUM_CLASSES, filterx=5, filtery=5, compiled=True),

    layers.GlobalAveragePooling(),
    layers.Activation("softmax"),
])
net = Backpropagation(layerstack=stack, cost="cxent", optimizer=optimizers.Adam(LEARNING_RATE))

net.fit_generator(stream.iter_subset("train", BATCH_SIZE),
                  lessons_per_epoch=stream.steps_per_epoch("train", BATCH_SIZE),
                  epochs=10,
                  metrics=["acc"],
                  validation=stream.iter_subset("val", BATCH_SIZE),
                  validation_steps=stream.steps_per_epoch("val", BATCH_SIZE))

# Save the weights as NumPy vector.
weights = net.get_weights(unfold=True)
np.save("AIM-SR-weights.npy", weights)
