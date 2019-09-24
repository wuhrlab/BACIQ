import inference_methods
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal as aae


def test_stan_single_init(mocker):
    mocker.patch('inference_methods.os.path.exists', return_value=True)
    mocker.patch('inference_methods.pickle.dump')
    mocker.patch('inference_methods.pickle.load',
                 return_value='Test Model')
    mock_open = mocker.patch('inference_methods.open')

    stan = inference_methods.Stan_Single('test')
    mock_open.assert_called_once_with('test', 'rb')
    assert stan.model == 'Test Model'

    mocker.patch('inference_methods.os.path.exists', return_value=False)
    mock_stan = mocker.patch('inference_methods.StanModel',
                             return_value='New Model')

    stan = inference_methods.Stan_Single('test')
    mock_open.assert_called_with('test', 'wb')
    assert stan.model == 'New Model'
    mock_stan.assert_called_once_with(model_name='baciq',
                                      model_code=mocker.ANY)


def test_stan_single_fit_quantiles(mocker):
    mocker.patch('inference_methods.os.path.exists', return_value=True)
    mocker.patch('inference_methods.open')
    mock_model = mocker.MagicMock()
    mock_model.sampling.return_value.extract.return_value = {
        'mu': range(101)}
    mocker.patch('inference_methods.pickle.load',
                 return_value=mock_model)
    stan = inference_methods.Stan_Single('test')

    data = pd.DataFrame({
        'ch0': [1, 2, 3],
        'sum': [10, 20, 30]
    })
    quants = stan.fit_quantiles(data, 'ch0', [0.1, 0.5, 0.9])
    aae(quants, np.array([10, 50, 90]))
    mock_model.sampling.assert_called_with(
        chains=1, iter=5005000, warmup=5000, refresh=-1,
        data={'J': 3,  # Number of peptides
              'y': data['ch0'],
              'n': data['sum']}
    )
