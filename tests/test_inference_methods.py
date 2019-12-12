from baciq import inference_methods
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal as aae


def test_get_num_cores(mocker):
    mocker.patch('baciq.inference_methods.os.environ', {})
    mocker.patch('baciq.inference_methods.multiprocessing.cpu_count',
                 return_value=-1)

    # without slurm env variable, get cpu_count
    assert inference_methods.get_num_cores() == -1

    mocker.patch('baciq.inference_methods.os.environ',
                 {'SLURM_CPUS_PER_TASK': '5'})
    # with slurm env variable, get its value as an int
    assert inference_methods.get_num_cores() == 5


@pytest.fixture
def pymc(mocker):
    mocker.patch('baciq.inference_methods.get_num_cores',
                 return_value='TESTING')
    mocker.patch('baciq.inference_methods.hist_backend.Histogram')
    return inference_methods.PYMC_Model(100, 2, 50, 'ch7')


def test_PYMC_Model_init(pymc):
    assert pymc.samples == 100
    assert pymc.chains == 2
    assert pymc.tuning == 50
    assert pymc.channel == 'ch7'
