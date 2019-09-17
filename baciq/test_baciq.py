from io import StringIO
import baciq
import pandas as pd
from pandas.testing import assert_frame_equal as afe
import pytest
from click.testing import CliRunner


@pytest.fixture
def file_gen():
    def _gen_file():
        return StringIO(
            'Protein ID,ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10\n'
            'BP677536,180.3139074,440.4864578,385.6522888,354.9392163,'
            '180.3139074,440.4864578,385.6522888,354.9392163,440.4864578,'
            '385.62,354.939\n'
            'BP677536,189.4696091,458.6363471,368.3842762,466.189303,'
            '189.4696091,458.6363471,368.384276,466.189303,458.6363471,'
            '368.28,446.18\n'
            'BP677537,189.4696091,458.6363471,368.3842762,466.189303,'
            '189.4696091,458.6363471,368.384276,466.189303,458.6363471,'
            '368.28,446.18\n'  # should be removed (only one protein)
        )
    return _gen_file


def test_read_df(file_gen):
    infile = file_gen()
    result = baciq.read_df(infile, 'ch0', 'ch1', 1)
    name, df = next(result)
    afe(df, pd.DataFrame({
        'Protein ID': ['BP677536', 'BP677536'],
        'ch0': [180, 189],
        'sum': [620, 648]
    }))
    with pytest.raises(StopIteration):
        next(result)

    infile = file_gen()
    result = baciq.read_df(infile, 'ch2', 'sum', 1.0/400)
    name, df = next(result)
    afe(df, pd.DataFrame({
        'Protein ID': ['BP677536', 'BP677536'],
        'ch2': [1, 1],
        'sum': [11, 11]
    }))
    with pytest.raises(StopIteration):
        next(result)


def test_read_df_with_non_numeric():
    infile = StringIO(
            'Protein ID,peptide,ch0,ch1\n'
            'BP677536,p1,180.3139074,440.4864578\n'
            'BP677536,p2,189.4696091,458.6363471\n'
    )
    result = baciq.read_df(infile, 'ch0', 'ch1', 1)
    name, df = next(result)
    afe(df, pd.DataFrame({
        'Protein ID': ['BP677536', 'BP677536'],
        'ch0': [180, 189],
        'sum': [620, 648]
    }))
    with pytest.raises(StopIteration):
        next(result)


def test_main(file_gen, mocker):
    stanmodel = mocker.patch('baciq.pystan.StanModel')
    stanmodel.return_value.sampling.return_value.extract.return_value = {
        'mu': list(range(101))}

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('infile.csv', 'w') as infile:
            infile.write(file_gen().getvalue())

        result = runner.invoke(
            baciq.main,
            '-c1 ch0 -c2 ch1 -i infile.csv -o outfile.csv'.split())
        assert result.exit_code == 0
        out = pd.read_csv('outfile.csv')
        afe(out, pd.DataFrame({
            'Protein ID': ['BP677536'],
            '0.025': [2.5],
            '0.5': [50],
            '0.975': [97.5],
        }))


def test_main_two_prot(file_gen, mocker):
    stanmodel = mocker.patch('baciq.pystan.StanModel')
    stanmodel.return_value.sampling.return_value.extract.side_effect = [
        {'mu': list(range(101))},
        {'mu': list(range(2, 202, 2))}
    ]

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('infile.csv', 'w') as infile:
            infile.write(
                'Protein ID,peptide,ch0,ch1\n'
                'BP677536,p1,180.3139074,440.4864578\n'
                'BP677536,p2,189.4696091,458.6363471\n'
                'BP677537,p1,180.3139074,440.4864578\n'
                'BP677537,p2,189.4696091,458.6363471\n'
            )

        result = runner.invoke(
            baciq.main,
            '-c1 ch0 -c2 ch1 -i infile.csv -o outfile.csv'.split())
        print(result)
        assert result.exit_code == 0
        out = pd.read_csv('outfile.csv')
        afe(out, pd.DataFrame({
            'Protein ID': ['BP677536', 'BP677537'],
            '0.025': [2.5, 6.95],
            '0.5': [50, 101],
            '0.975': [97.5, 195.05],
        }))
