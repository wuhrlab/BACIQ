from io import StringIO
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
    result = BB_code.read_df(1, 1, 1, 1, 0, 1, infile)
    name, df = next(result)
    assert name == 'BP677536'
    afe(df, pd.DataFrame({
        'Protein ID': ['BP677536', 'BP677536'],
        'channel1': [180, 189],
        'channel2': [440, 459]
    }))
    with pytest.raises(StopIteration):
        next(result)

    infile = file_gen()
    result = BB_code.read_df(1.0/400, 1, 1, 0, 2, 1, infile)
    name, df = next(result)
    afe(df, pd.DataFrame({
        'Protein ID': ['BP677536', 'BP677536'],
        'channel1': [1, 1],
        'channel2': [10, 10]
    }))
    with pytest.raises(StopIteration):
        next(result)
