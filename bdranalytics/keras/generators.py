import numpy as np
from keras.preprocessing.image import Iterator


class StratifiedIndexGenerator:
    """
    Stratified index generator.
    """

    def __init__(self, shuffle=True):
        self.shuffle = shuffle

    def flow(self, strata=None, batch_size=128, strata_weights=None):
        """
        Batch generator function.

        :param strata: vector with n_samples that denotes the strata
        :param batch_size: number of samples in a batch
        :param strata_weights: dictionary with strata weights, should sum to 1.0
        :return: an iterator that yields sample indices
        """

        strata = strata.ravel()
        strata_labels = np.unique(strata)
        n_strata = len(strata_labels)
        n_samples_total = len(strata)  # total number of samples
        indices = np.arange(n_samples_total)

        generators = {}
        surplus = {}

        if strata_weights is None:
            strata_weights = {}

        for stratum_label in strata_labels:
            mask = (strata == stratum_label)
            generators[stratum_label] = self.sample_generator(indices[mask])
            surplus[stratum_label] = 0.
            if strata_weights.get(stratum_label) is None:
                strata_weights[stratum_label] = 1 / n_strata

        if sum(strata_weights.values()) != 1.0:
            raise ValueError("strata weights should sum to 1.0")

        # preallocate
        return_indices = np.empty(batch_size, dtype=int)

        while True:

            # shift labels array to make sure every label is last label equal amount of times
            # better dispersed surplus
            strata_labels = np.roll(strata_labels, 1)

            # reset total sample counter
            i_sample = 0
            for stratum_label in strata_labels:

                # float indicating number of samples to draw from this stratum
                n_samples_float = (
                    batch_size * strata_weights[stratum_label]) - surplus[stratum_label]

                # exception when reaching last stratum
                if stratum_label == strata_labels[-1]:
                    n_samples = batch_size - i_sample
                else:
                    n_samples = round(n_samples_float)

                # store remainder
                surplus[stratum_label] = (1. * n_samples) - n_samples_float

                if n_samples == 0:
                    continue

                # draw samples from generator
                for _ in range(n_samples):
                    return_indices[i_sample] = next(generators[stratum_label])
                    i_sample += 1  # increment total sample counter

            # yield result
            yield return_indices

    def sample_generator(self, indices):
        """
        Basic single element generator from a list, in shuffled order.

        :param indices: list of indices to yield
        :return: a generator
        """
        while True:
            if self.shuffle:
                indices = np.random.permutation(indices)
            for selected_row in indices:
                yield selected_row


class DataGenerator:
    """
    Keras-API compatible data generator class for in-memory (X, y) samples.
    Comparable to keras.preprocessing.image.ImageDataGenerator
    """

    def __init__(self):
        pass

    def flow(self, X, y, batch_size=128, seed=42, shuffle=True, strata=None, strata_weights=None):
        """
        Returns a data iterator that can be looped over to return batches.

        :param X: array-like, input data
        :param y: array-like, target data
        :param batch_size: int, number of samples in the batch
        :param seed: int, seed for randomness, set globally
        :param shuffle: bool, whether to shuffle the dataset
        :param strata: array-like, size n_samples that denotes the subpopulation (stratum) ID, which
                        is sampled independently.
        :param strata_weights: dictionary, containing strata weights, should sum to 1.0
        :return: an iterator
        """
        return DataIterator(X, y, batch_size=batch_size, n=X.shape[0], seed=seed, shuffle=shuffle,
                            strata=strata, strata_weights=strata_weights)


class DataIterator(Iterator):
    """
    Data iterator stratification capability. Keras-API compatible.
    Comparable to keras.preprocessing.image.NumpyDataIterator
    """

    def __init__(self, X, y, strata=None, strata_weights=None, batch_size=128, shuffle=True, **kwargs):
        self.X = X
        self.y = y
        self.strata = strata
        self.strata_weights = strata_weights

        super(DataIterator, self).__init__(
            batch_size=batch_size, shuffle=shuffle, **kwargs)

        if self.strata is not None:
            self.index_generator = StratifiedIndexGenerator(shuffle=shuffle).flow(batch_size=batch_size,
                                                                                  strata=self.strata,
                                                                                  strata_weights=self.strata_weights)

    def _get_batches_of_transformed_samples(self, index_array):
        return self.X[index_array, ], self.y[index_array, ]

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
