from baciq import inference_methods
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal as aae
from numpy.testing import assert_allclose as aac


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


def test_get_proteins_and_indices():
    proteins, idx = inference_methods.get_proteins_and_indices(
        pd.DataFrame(
            {'Protein ID': 'a b c d d a a'.split()}
        ))
    aae(proteins, 'a b c d'.split())
    aae(idx, [0, 1, 2, 3, 3, 0, 0])

    proteins, idx = inference_methods.get_proteins_and_indices(
        pd.DataFrame(
            {'Protein ID': 'd c b a d a a'.split()}
        ))
    aae(proteins, 'd c b a'.split())
    aae(idx, [0, 1, 2, 3, 0, 3, 3])

    with pytest.raises(KeyError) as e:
        inference_methods.get_proteins_and_indices(pd.DataFrame())
    assert 'Expected "Protein ID" in input data' in str(e.value)

    proteins, idx = inference_methods.get_proteins_and_indices(
        pd.DataFrame(
            {'Protein ID': []}
        ))
    aae(proteins, [])
    aae(idx, [])


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


def test_PYMC_Model_mcmc_sample(pymc, mocker):
    mock_mc = mocker.patch('baciq.inference_methods.pm')
    proteins, result = pymc.mcmc_sample(
        pd.DataFrame({
            'Protein ID': 'a b c d a'.split(),
            'ch7': [1, 2, 3, 4, 5],
            'sum': [10, 11, 12, 13, 14]
        }),
        bin_width=0.2)

    aae(proteins, 'a b c d'.split())
    # this checks all the params for the pymc model
    mock_mc.sample.assert_called_with(
        chains=2, compute_convergence_checks=False, cores='TESTING',
        draws=100, progressbar=False, trace=mocker.ANY, tune=50)
    mock_mc.Gamma.assert_called_with(
        'τ', alpha=7.5, beta=1)
    mock_mc.Bound.assert_called_with(
        mocker.ANY, lower=0, upper=1)
    mock_mc.Bound.return_value.assert_called_with(
        'μ', mu=0.5, shape=4, sigma=1)
    mock_mc.Exponential.assert_called_with(
        'κ', mock_mc.Gamma.return_value, shape=4)
    mock_mc.BetaBinomial.assert_called_with(
        'y', alpha=mocker.ANY, beta=mocker.ANY,
        n=mocker.ANY, observed=mocker.ANY)

    assert (mock_mc.BetaBinomial.call_args[1]['n'] ==
            [10, 11, 12, 13, 14]).all()
    assert (mock_mc.BetaBinomial.call_args[1]['observed'] ==
            [1, 2, 3, 4, 5]).all()


def test_PYMC_Model_unmocked_mcmc_sample():
    pymc = inference_methods.PYMC_Model(1000, 2, 500, 'ch0')
    proteins, result = pymc.mcmc_sample(
        pd.DataFrame({
            'Protein ID': 'b d c a'.split(),
            'ch0': [10, 250, 500, 1000],
            'sum': [1000, 1000, 1000, 1000]}),
        bin_width=0.1)

    aae(proteins, 'b d c a'.split())
    aae(result.sum(axis=1), [2000, 2000, 2000, 2000])  # 1000 samples * 2 chain
    bins = np.array(range(10))/10
    aac(np.sum(result * bins[None, :] / 2000, axis=1),
        [0.38, 0.43, 0.45, 0.66], atol=0.05)

    proteins, result = pymc.mcmc_sample(
        pd.DataFrame({
            'Protein ID': ['b', 'd', 'c', 'a']*10,
            'ch0': [10, 250, 500, 750]*10,
            'sum': [1000, 1000, 1000, 1000]*10}),
        bin_width=0.1)

    aae(proteins, 'b d c a'.split())
    aae(result.sum(axis=1), [2000, 2000, 2000, 2000])  # 1000 samples * 2 chain
    aac(np.sum(result * bins[None, :] / 2000, axis=1),
        [0, 0.2, 0.45, 0.7], atol=0.05)  # this is low because of binning


def test_PYMC_Model_fit_histogram(pymc, mocker):
    mock_sample = mocker.patch.object(
        inference_methods.PYMC_Model, 'mcmc_sample',
        return_value=('a b c'.split(),
                      np.array([
                          [0, 10, 20, 10, 0],
                          [40, 0, 0, 0, 0],
                          [0, 0, 0, 30, 10]
                      ])))
    result = pymc.fit_histogram(pd.DataFrame({
        'Protein ID': 'a b c'.split()}),  # doesn't matter
                                bin_width=0.2)
    mock_sample.assert_called_with(mocker.ANY, 0.2)

    # because pandas df equality is infuriating!
    assert (result.index == 'a b c'.split()).all()
    assert result.index.name == 'Protein ID'
    aac(result.columns.values, [0, 0.2, 0.4, 0.6, 0.8])
    assert (result.values ==
            [
                [0, 10, 20, 10, 0],
                [40, 0, 0, 0, 0],
                [0, 0, 0, 30, 10]
            ]).all()


def test_PYMC_Model_fit_quantiles(pymc, mocker):
    return_quants = np.zeros((3, 10000))
    return_quants[0, 1000:1101] = 1  # 100 1's
    return_quants[1, 0] = 1  # first index
    return_quants[2, 9999] = 1  # last index
    mock_sample = mocker.patch.object(
        inference_methods.PYMC_Model, 'mcmc_sample',
        return_value=('a c b'.split(), return_quants))
    result = pymc.fit_quantiles(pd.DataFrame({
        'Protein ID': 'a b c'.split()}),  # doesn't matter
                                [0.025, 0.5, 0.975])
    assert mock_sample.call_args[1]['bin_width'] == 0.0001
    assert (result.index == 'a c b'.split()).all()
    assert result.index.name == 'Protein ID'
    assert (result.columns.values == ['0.025', '0.5', '0.975']).all()
    print(result)
    aac(result.values,
        [
            [0.1002, 0.1050, 0.1098],
            [0, 0, 0],
            [0.9999, 0.9999, 0.9999],
        ])
