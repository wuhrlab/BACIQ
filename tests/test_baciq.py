from io import StringIO
from baciq import baciq
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal as aae
import pytest
from click.testing import CliRunner


@pytest.fixture
def file_gen():
    def _gen_file():
        return StringIO(
            'Protein ID,ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,pep\n'
            'BP677536,180.3139074,440.4864578,385.6522888,354.9392163,'
            '180.3139074,440.4864578,385.6522888,354.9392163,440.4864578,'
            '385.62,354.939,a\n'
            'BP677536,189.4696091,458.6363471,368.3842762,466.189303,'
            '189.4696091,458.6363471,368.384276,466.189303,458.6363471,'
            '368.28,446.18,b\n'
            'BP677537,189.4696091,458.6363471,368.3842762,466.189303,'
            '189.4696091,458.6363471,368.384276,466.189303,458.6363471,'
            '368.28,446.18,c\n'
        )
    return _gen_file


def check_df(expected, result, decimals=None):
    # check all columns in result
    for col in expected.columns.values:
        assert col in result.keys()
    # check all items in result match expected
    for k, v in result.items():
        if decimals is None or k == 'Protein ID':
            assert (expected[k] == v).all()
        else:
            aae(np.round(expected[k], decimals), np.round(v, decimals))


def test_read_df(file_gen):
    # test simple
    infile = file_gen()
    result = baciq.read_df(infile, 'ch0', 'ch1', 1)
    df = next(result)
    check_df(df, {
        'Protein ID': ['BP677536', 'BP677536', 'BP677537'],
        'ch0': [180, 189, 189],
        'sum': [620, 648, 648]
    })
    with pytest.raises(StopIteration):
        next(result)

    # test scaling
    infile = file_gen()
    result = baciq.read_df(infile, 'ch2', 'sum', 1.0/400)
    df = next(result)
    check_df(df, {
        'Protein ID': ['BP677536', 'BP677536', 'BP677537'],
        'ch2': [1, 1, 1],
        'sum': [11, 11, 11]
    })
    with pytest.raises(StopIteration):
        next(result)

    # test batch size
    infile = file_gen()
    np.random.seed(0)
    result = baciq.read_df(infile, 'ch2', 'sum', 1.0/400, batch_size=1)
    df = next(result)
    check_df(df, {
        'Protein ID': ['BP677537'],
        'ch2': [1],
        'sum': [11]
    })
    df = next(result)
    check_df(df, {
        'Protein ID': ['BP677536', 'BP677536'],
        'ch2': [1, 1],
        'sum': [11, 11]
    })
    with pytest.raises(StopIteration):
        next(result)

    # test batch size with odd protein number
    np.random.seed(2)
    result = baciq.read_df(StringIO(
        'Protein ID,ch0,ch1\n'
        'pro1,1,2\n'
        'pro2,1,2\n'
        'pro6,1,2\n'
        'pro7,1,2\n'
        'pro3,1,2\n'
        'pro4,1,2\n'
        'pro5,1,2\n'
        'pro1,1,2\n'
    ), 'ch0', 'ch1', 1, batch_size=3)
    # NOTE the order is retained from the original for each batch
    df = next(result)
    check_df(df, {
        'Protein ID': ['pro2', 'pro4', 'pro5'],
        'ch0': [1, 1, 1],
        'sum': [3, 3, 3]
    })

    df = next(result)
    check_df(df, {
        'Protein ID': ['pro7', 'pro3'],
        'ch0': [1, 1],
        'sum': [3, 3]
    })

    df = next(result)
    check_df(df, {
        'Protein ID': ['pro1', 'pro6', 'pro1'],
        'ch0': [1, 1, 1],
        'sum': [3, 3, 3]
    })
    with pytest.raises(StopIteration):
        next(result)


infile_str = (
    'Protein ID,peptide,ch0,ch1\n'
    'BP677536,p1,180.3139074,440.4864578\n'
    'BP677536,p2,189.4696091,458.6363471\n'
    'BP677537,p1,180.3139074,440.4864578\n'
    'BP677537,p2,189.4696091,458.6363471\n'
)


def test_main_two_prot_defaults_unmocked(file_gen):
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('infile.csv', 'w') as infile:
            infile.write(infile_str)

        np.random.seed(0)
        result = runner.invoke(
            baciq.main,
            '-c1 ch0 -c2 ch1 -i infile.csv -o outfile.csv'.split())
        assert result.exit_code == 0
        assert 'Estimating quantiles' in result.output
        out = pd.read_csv('outfile.csv')
        check_df(out, {
            'Protein ID': ['BP677536', 'BP677537'],
            '0.025': [0.1439, 0.1420],
            '0.5': [0.4613, 0.4596],
            '0.975': [0.8250, 0.8227],
        }, decimals=1)

        # binned
        np.random.seed(0)
        result = runner.invoke(
            baciq.main,
            '-c1 ch0 -c2 ch1 -i infile.csv -o outfile.csv -b 0.1'.split())
        assert result.exit_code == 0
        assert 'Estimating histogram' in result.output
        out = pd.read_csv('outfile.csv')
        # values are machine dependent it seems like
        assert (out['Protein ID'] == ['BP677536', 'BP677537']).all()
        assert out.values.shape == (2, 11)
        aae(out.values[:, 1:].sum(axis=1), [5000, 5000])


def test_main_quants(mocker):
    quants = pd.DataFrame({
        'Protein ID': 'a b c'.split(),
        '0.025': [0.1, 0.2, 0.3],
        '0.5': [0.4, 0.5, 0.6],
        '0.975': [0.7, 0.8, 0.9]
    }).set_index('Protein ID')
    mock_pymc = mocker.patch('baciq.inference_methods.PYMC_Model')
    mock_pymc.return_value.fit_quantiles.side_effect = [quants]

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('infile.csv', 'w') as infile:
            infile.write(infile_str)

        np.random.seed(0)
        result = runner.invoke(
            baciq.main,
            '-c1 ch0 -c2 ch1 -i infile.csv -o outfile.csv'.split())
        assert result.exit_code == 0
        assert 'Estimating quantiles' in result.output
        mock_pymc.assert_called_with(1000, 5, 1000, 'ch0')
        assert mock_pymc.return_value.fit_quantiles.call_args[0][1] == \
            pytest.approx([0.025, 0.5, 0.975])
        with open('outfile.csv') as reader:
            assert reader.readlines() == [
                'Protein ID,0.025,0.5,0.975\n',
                'a,0.1,0.4,0.7\n',
                'b,0.2,0.5,0.8\n',
                'c,0.3,0.6,0.9\n'
            ]


def test_main_quants_nondefault(mocker):
    quants = pd.DataFrame({
        'Protein ID': 'a b c'.split(),
        '0.05': [0.1, 0.2, 0.3],
        '0.5': [0.4, 0.5, 0.6],
        '0.95': [0.7, 0.8, 0.9]
    }).set_index('Protein ID')
    quants2 = pd.DataFrame({
        'Protein ID': 'd'.split(),
        '0.05': [0.1],
        '0.5': [0.4],
        '0.975': [0.7]  # column won't matter
    }).set_index('Protein ID')
    mock_pymc = mocker.patch('baciq.inference_methods.PYMC_Model')
    mock_pymc.return_value.fit_quantiles.side_effect = [quants, quants2]

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('infile.csv', 'w') as infile:
            infile.write(infile_str)

        np.random.seed(0)
        result = runner.invoke(
            baciq.main,
            ('-c1 ch0 -c2 ch1 -i infile.csv -o outfile.csv '
             '-c 0.9 --samples 500 --chains 2 --tuning 250 '
             '--batch-size 1').split())
        assert result.exit_code == 0
        assert ('Estimating quantiles\nBatch 1 of 2. 1 proteins\n'
                'Batch 2 of 2. 1 proteins\n') == result.output
        mock_pymc.assert_called_with(500, 2, 250, 'ch0')
        assert mock_pymc.return_value.fit_quantiles.call_args[0][1] == \
            pytest.approx([0.05, 0.5, 0.95])
        with open('outfile.csv') as reader:
            assert reader.readlines() == [
                'Protein ID,0.05,0.5,0.95\n',
                'a,0.1,0.4,0.7\n',
                'b,0.2,0.5,0.8\n',
                'c,0.3,0.6,0.9\n',
                'd,0.1,0.4,0.7\n'
            ]


def test_main_hist_nondefault(mocker):
    quants = pd.DataFrame({
        'Protein ID': 'a b c'.split(),
        '0.0': [1, 2, 3],
        '0.5': [4, 5, 6],
    }).set_index('Protein ID')
    quants2 = pd.DataFrame({
        'Protein ID': 'd'.split(),
        '0.05': [1],  # column won't matter
        '0.5': [4],
    }).set_index('Protein ID')
    mock_pymc = mocker.patch('baciq.inference_methods.PYMC_Model')
    mock_pymc.return_value.fit_histogram.side_effect = [quants, quants2]

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('infile.csv', 'w') as infile:
            infile.write(infile_str)

        np.random.seed(0)
        result = runner.invoke(
            baciq.main,
            ('-c1 ch0 -c2 ch1 -i infile.csv -o outfile.csv '
             '-c 0.9 --samples 500 --chains 2 --tuning 250 '  # c is ignored
             '--batch-size 1 --bin-width 0.5').split())
        assert result.exit_code == 0
        assert ('Estimating histogram\nBatch 1 of 2. 1 proteins\n'
                'Batch 2 of 2. 1 proteins\n') == result.output
        mock_pymc.assert_called_with(500, 2, 250, 'ch0')
        assert mock_pymc.return_value.fit_histogram.call_args[0][1] == \
            pytest.approx(0.5)
        with open('outfile.csv') as reader:
            assert reader.readlines() == [
                'Protein ID,0.0,0.5\n',
                'a,1,4\n',
                'b,2,5\n',
                'c,3,6\n',
                'd,1,4\n'
            ]
