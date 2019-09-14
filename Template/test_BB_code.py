from io import StringIO, BytesIO
import BB_code
import pandas as pd
from pandas.testing import assert_frame_equal as afe
import pytest


@pytest.fixture
def file_gen():
    def _gen_file():
        return StringIO(
            u'Protein ID,ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10\n'
            u'BP677536,180.3139074,440.4864578,385.6522888,354.9392163,'
            u'180.3139074,440.4864578,385.6522888,354.9392163,440.4864578,'
            u'385.62,354.939\n'
            u'BP677536,189.4696091,458.6363471,368.3842762,466.189303,'
            u'189.4696091,458.6363471,368.384276,466.189303,458.6363471,'
            u'368.28,446.18\n'
            u'BP677537,189.4696091,458.6363471,368.3842762,466.189303,'
            u'189.4696091,458.6363471,368.384276,466.189303,458.6363471,'
            u'368.28,446.18\n'  # should be removed (only one protein)
        )
    return _gen_file


def test_read_df(file_gen):
    infile = file_gen()
    result = BB_code.read_df(infile, 'ch0', 'ch1', 1)
    name, df = next(result)
    afe(df, pd.DataFrame({
        'Protein ID': ['BP677536', 'BP677536'],
        'ch0': [180, 189],
        'sum': [620, 648]
    }))
    with pytest.raises(StopIteration):
        next(result)

    infile = file_gen()
    result = BB_code.read_df(infile, 'ch2', 'sum', 1.0/400)
    name, df = next(result)
    afe(df, pd.DataFrame({
        'Protein ID': ['BP677536', 'BP677536'],
        'ch2': [1, 1],
        'sum': [11, 11]
    }))
    with pytest.raises(StopIteration):
        next(result)


def test_main(file_gen, mocker):
    stanmodel = mocker.patch('BB_code.pystan.StanModel')
    stanmodel.return_value.sampling.return_value.extract.return_value = {
        'mu': list(range(101))}
    mocker.patch('BB_code.sys.argv', ['', 'ch0', 'ch1', 1, 0.95])
    infile = file_gen()
    output = BytesIO()

    BB_code.main(infile, output)
    output.seek(0)
    out = pd.read_csv(output)
    afe(out, pd.DataFrame({
        'index': ['BP677536'],
        '0.025': [2.5],
        '0.5': [50],
        '0.975': [97.5]
    }))
