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
    # BrainForge is channels first, but I did the architecture search in a GPU-enabled lib.
    CHANNELS_FIRST = "channels_first"
    CHANNELS_LAST = "channels_last"


class Dataset:

    """
    Streamer and parser for the 52x52 road sign classification dataset
    """

    DEFAULT_ROOT = "data/train-52x52"
    NUM_CLASSES = 12

    def __init__(self,
                 root: str = "default",
                 split_validation: float = 0.2,
                 image_format: str = Format.CHANNELS_FIRST):

        """
        :param root: str
            The data root directory.
        :param split_validation: float
            Validation set will be split from the end of every directory.
        :param image_format:
            Whether we use channels-first or channels-last representation.
        """

        if root == "default":
            root = self.DEFAULT_ROOT

        self.root = pathlib.Path(root)

        # We'll store all image paths and labels together.
        self.paths = []
        self.labels = []

        # Only their indices will be split.
        self.subset_indices = {Subset.TRAIN: [], Subset.VAL: []}

        self.format = image_format
        self.split_validation = split_validation

        self._parse_data_root()

    def _parse_data_root(self):

        if not self.root.exists():
            raise RuntimeError("Supplied dataset root does not exist: {}".format(str(self.root)))

        directories = sorted(self.root.glob("*"), key=lambda path: str(path))  # Sorting for reproducibility

        for directory in directories:  # iterate over the class-directories

            if not directory.is_dir():  # required to skip the pesky dataset.png file
                continue

            label = os.path.split(directory)[-1]  # infer the label from the directory name
            label = int(label) - 1  # labels are indexed from 1, but we'll index them from 0

            files = sorted(list(directory.glob("*.bmp")), key=lambda path: str(path))  # Sorting for reproducibility

            to_train = len(files) - int(len(files) * self.split_validation)

            for j, file_path in enumerate(files):

                self.paths.append(file_path)
                self.labels.append(label)  # labels are the same for every file in a directory

                sample_id = len(self.paths) - 1
                subset = Subset.TRAIN if j < to_train else Subset.VAL
                self.subset_indices[subset].append(sample_id)

        self.labels = np.array(self.labels)  # easier to index later

        print(" [Streamer] - Num train samples:", len(self.subset_indices[Subset.TRAIN]))
        print(" [Streamer] - Num val samples:", len(self.subset_indices[Subset.VAL]))

        assert len(np.unique(self.labels)) == self.NUM_CLASSES

    @property
    def input_shape(self):
        if self.format == Format.CHANNELS_FIRST:
            return [3, 52, 52]
        elif self.format == Format.CHANNELS_LAST:
            return [52, 52, 3]
        assert False, "w00t"

    def steps_per_epoch(self, subset: str, batch_size: int):
        return len(self.subset_indices[subset]) // batch_size

    def _read_and_preprocess(self, image_id):
        image = cv2.imread(str(self.paths[image_id])) / 255.  # scale to [0. - 1.] range
        if self.format == Format.CHANNELS_FIRST:
            image = image.transpose((2, 0, 1))
        return image

    def iter_subset(self, subset: str, batch_size: int):
        """
        Creates an iterator, which can be iterated for batch_size samples.
        Yields tuples of network inputs (downscaled and appropriately formatted images)
        and one-hot encoded class labels from the subset specified.

        :param subset: str
            Which subset to create the iterator from .
        :param batch_size: int
            Minibatch size for Stochastic Gradient Descent training.
        """

        indices = self.subset_indices[subset]

        if len(indices) < batch_size:
            raise RuntimeError(f"Too few samples in subset: {subset}. Found only {len(indices)} samples.")

        batch_x = np.empty([batch_size] + self.input_shape, dtype="float64")
        batch_y = np.empty([batch_size, self.NUM_CLASSES], dtype="float64")

        while 1:
            idx = np.random.choice(indices, size=batch_size, replace=False)

            # Set the label tensor
            batch_y[:] = 0.
            batch_y[range(batch_size), self.labels[idx]] = 1.  # sets all the one-hots at once

            # Set the input tensor
            for i, ID in enumerate(idx):
                batch_x[i] = self._read_and_preprocess(ID)

            yield batch_x, batch_y
