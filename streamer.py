"""
This module contains the logic for data parsing and streaming
"""

import os
import pathlib

import numpy as np
import cv2


class Subset:
    # I don't like Python Enums.
    TRAIN = "train"
    VAL = "val"


class Format:
    # Brainforge is channels first, but I did the hyperparam search in Keras :)
    CHANNELS_FIRST = "channels_first"
    CHANNELS_LAST = "channels_last"


class Stream:

    """
    Streamer and parser for the 52x52 road sign classification dataset
    """

    DEFAULT_ROOT = "data/train-dataset-52x52"
    NUM_CLASSES = 12

    def __init__(self,
                 root: str = "default",
                 split_validation: float = 0.2,
                 image_format: str = Format.CHANNELS_FIRST):

        """
        :param root: str
            The data root directory.
        :param split_validation: float
            Split factor for the validation set.
        :param image_format:
            Whether we use channels-first or channels-last representation.
        """

        if root == "default":
            root = self.DEFAULT_ROOT

        self.root = pathlib.Path(root)
        self.paths = []
        self.labels = []

        self.subset_indices = {Subset.TRAIN: [], Subset.VAL: []}

        for i, lib in enumerate(self.root.glob("*")):
            label = os.path.split(lib)[-1]
            if label == "dataset.png":
                continue
            label = int(label) - 1
            files = list(lib.glob("*.bmp"))
            to_train = len(files) - int(len(files) * split_validation)
            for j, file_path in enumerate(files):
                self.paths.append(file_path)
                self.labels.append(label)
                subset = Subset.TRAIN if j < to_train else Subset.VAL
                self.subset_indices[subset].append(len(self.paths)-1)

        print(" [Streamer] - Num train samples:", len(self.subset_indices[Subset.TRAIN]))
        print(" [Streamer] - Num val samples:", len(self.subset_indices[Subset.VAL]))
        self.format = image_format

    @property
    def input_shape(self):
        if self.format == Format.CHANNELS_FIRST:
            return [3, 52, 52]
        elif self.format == Format.CHANNELS_LAST:
            return [52, 52, 3]
        assert False

    def steps_per_epoch(self, subset: str, batch_size: int):
        return len(self.subset_indices[subset]) // batch_size

    def iter_subset(self, subset: str, batch_size: int):
        indices = self.subset_indices[subset]

        batch_x = np.empty([batch_size] + self.input_shape, dtype="float64")
        batch_y = np.empty([batch_size, self.NUM_CLASSES], dtype="float64")

        while 1:
            idx = np.random.choice(indices, size=batch_size)
            batch_y[:] = 0.
            for i, ID in enumerate(idx):
                image = cv2.imread(str(self.paths[ID])) / 255.
                if self.format == Format.CHANNELS_FIRST:
                    image = image.transpose((2, 0, 1))
                batch_x[i] = image
                batch_y[i, self.labels[ID]] = 1.

            yield batch_x, batch_y
