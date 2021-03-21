from baciq import hist_backend
import pymc3 as pm
import pytest
import numpy as np
from numpy.testing import assert_array_equal as aae


@pytest.fixture(name='model')
def pm_model():
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=1)
        sigma = pm.Normal('sigma', mu=0, sigma=1)
        pm.Normal('obs', mu=mu, sigma=sigma)
        return model


def test_init(model):
    with model:
        hist = hist_backend.Histogram(vars=[model.mu])

    assert hist.name == 'histogram'
    assert hist.bin_width == 0.1
    assert hist.hist == {}
    assert hist.rows == {}
    assert hist.sample_count == {}
    assert hist.remove_first == 0
    assert hist.varnames == ['mu']

    with model:
        hist = hist_backend.Histogram(vars=[model.obs],
                                      bin_width=0.01, remove_first=100)

    assert hist.bin_width == 0.01
    assert hist.hist == {}
    assert hist.rows == {}
    assert hist.sample_count == {}
    assert hist.remove_first == 100
    assert hist.varnames == ['obs']

    with model:
        with pytest.raises(ValueError) as e:
            hist_backend.Histogram(vars=[model.obs], bin_width=0.3)

        assert 'Provided bin_width does not evenly divide range of (0,1)' in\
            str(e.value)


def test_setup(model):
    with model:
        hist = hist_backend.Histogram(vars=[model.mu])

    hist.setup(100, 0)
    assert hist.chain == 0
    assert hist.sample_count == {0: 0}
    hist.setup(1000, 1)
    assert hist.chain == 1
    assert hist.sample_count == {0: 0, 1: 0}


def test_record(model):
    with model:
        hist = hist_backend.Histogram(vars=[model.mu])
        assert len(hist) == 0
        hist.setup(100, 0)
        hist.record({'mu': np.array(0.1), 'sigma': None, 'obs': None})
        assert hist.sample_count[0] == 1
        assert len(hist) == 1
        aae(hist.hist['mu'],
            np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]))
        aae(hist.rows['mu'],
            np.array([0]))
        hist.record({'mu': np.array(0.1), 'sigma': None, 'obs': None})
        hist.record({'mu': np.array(0.2), 'sigma': None, 'obs': None})
        hist.record({'mu': np.array(0), 'sigma': None, 'obs': None})
        hist.record({'mu': np.array(1), 'sigma': None, 'obs': None})
        hist.record({'mu': np.array(0.99), 'sigma': None, 'obs': None})
        assert hist.sample_count[0] == 6
        assert len(hist) == 6
        aae(hist.hist['mu'],
            np.array([[1, 2, 1, 0, 0, 0, 0, 0, 0, 2]]))
        with pytest.raises(ValueError) as e:
            hist.record({'mu': np.array(1.001), 'sigma': None, 'obs': None})
        assert ('Histogram backend only supports values in [0, 1], '
                '"mu" had value of 1.001') in str(e.value)
        with pytest.raises(ValueError) as e:
            hist.record({'mu': np.array(-1), 'sigma': None, 'obs': None})
        assert ('Histogram backend only supports values in [0, 1], '
                '"mu" had value of -1.0') in str(e.value)
        with pytest.raises(ValueError) as e:
            hist.record({'mu': np.array(-0.5), 'sigma': None, 'obs': None})
        assert ('Histogram backend only supports values in [0, 1], '
                '"mu" had value of -0.5') in str(e.value)

    with pm.Model():
        mu = pm.Normal('mu', mu=0, sigma=1, shape=2)
        sigma = pm.Normal('sigma', mu=0, sigma=1, shape=2)
        pm.Normal('obs', mu=mu, sigma=sigma, shape=2)
        hist = hist_backend.Histogram(vars=[mu, sigma],
                                      remove_first=1, bin_width=0.2)
        hist.setup(100, 1)

        hist.record({'mu': np.array([0.1, 0.2]), 'sigma': np.array([0, 0]),
                     'obs': None})
        assert hist.sample_count[1] == 1
        assert hist.hist == {}  # ignoring first sample
        assert len(hist) == 0

        hist.record({'mu': np.array([0.1, 0.2]), 'sigma': np.array([0, 0]),
                     'obs': None})
        assert hist.sample_count[1] == 2
        aae(hist.hist['mu'],
            np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]]))
        aae(hist.rows['mu'],
            np.array([0, 1]))

        aae(hist.hist['sigma'],
            np.array([[1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0]]))
        aae(hist.rows['sigma'],
            np.array([0, 1]))
        assert len(hist) == 1

        hist.record({'mu': np.array([0.3, 0.4]), 'sigma': np.array([1, 1]),
                     'obs': None})
        assert hist.sample_count[1] == 3
        aae(hist.hist['mu'],
            np.array([[1, 1, 0, 0, 0],
                      [0, 1, 1, 0, 0]]))

        aae(hist.hist['sigma'],
            np.array([[1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1]]))
        assert len(hist) == 2

        with pytest.raises(ValueError) as e:
            hist.record({'mu': np.array([0.3, -0.5]),
                         'sigma': np.array([1, 1]),
                         'obs': None})
        assert ('Histogram backend only supports values in [0, 1], '
                '"mu" had value of -0.5') in str(e.value)

        with pytest.raises(ValueError) as e:
            hist.record({'mu': np.array([0.3, 0.5]),
                         'sigma': np.array([1.01, 1]),
                         'obs': None})
        assert ('Histogram backend only supports values in [0, 1], '
                '"sigma" had value of 1.01') in str(e.value)

        assert hist[0:1] == hist  # slice is minimally supported
