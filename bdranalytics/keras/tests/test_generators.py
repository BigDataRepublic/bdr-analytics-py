import pytest
from bdranalytics.keras.generators import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


@pytest.yield_fixture(scope='class')
def params(request):
    request.cls.batch_size = 3
    request.cls.X = np.random.rand(100, 10)
    request.cls.X_image = np.random.rand(100, 1, 1, 1)
    request.cls.y = 1. * (np.random.rand(100, 1) > 0.5)
    request.cls.strata_weights = {1.0: 0.2, 0.0: 0.8}
    yield


@pytest.mark.usefixtures("params")
class TestGenerators:

    def test_stratified_index_generator(self):
        iterator = StratifiedIndexGenerator().flow(
            strata=self.y, batch_size=self.batch_size,
            strata_weights=self.strata_weights
        )
        total, positives = 0, 0

        for i in range(100):
            indices = next(iterator)
            positives += self.y[indices].sum()
            total += len(indices)

        np.testing.assert_almost_equal(
            np.array([positives / total]), np.array([0.2])
        )

    def test_data_generator(self):
        iterator = DataGenerator().flow(
            self.X, self.y, strata=self.y, strata_weights=self.strata_weights)

        total, positives = 0, 0

        for i in range(100):
            X, y = next(iterator)
            positives += y.sum()
            total += len(y)

        np.testing.assert_almost_equal(
            np.array([positives / total]), np.array([0.2])
        )

        model = Sequential()
        model.add(Dense(units=1, input_shape=self.X.shape[1:]))
        model.compile(loss='mean_squared_error', optimizer='sgd')
        model.fit_generator(
            iterator, steps_per_epoch=(len(self.X)/self.batch_size))

    def test_stratified_image_data_generator(self):

        iterator = ImageDataGenerator().flow(self.X_image, self.y)

        iterator.index_generator = StratifiedIndexGenerator().flow(
            batch_size=self.batch_size,
            strata=self.y,
            strata_weights=self.strata_weights)

        total, positives = 0, 0

        for i in range(100):
            X, y = next(iterator)
            positives += y.sum()
            total += len(y)

        np.testing.assert_almost_equal(
            np.array([positives / total]), np.array([0.2])
        )
