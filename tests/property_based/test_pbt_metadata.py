import pytest

import string

from hypothesis import given, settings, seed
import hypothesis.strategies as st


st_valid_names = st.text(min_size=1, alphabet=string.ascii_letters + string.digits + '_-.', max_size=32)
st_valid_ints = st.integers(min_value=0, max_value=999999)
st_valid_keys = st.one_of(st_valid_ints, st_valid_names)

st_valid_values = st.text(min_size=1, alphabet=string.printable + string.whitespace, max_size=4_000)


@settings(max_examples=100, deadline=3000.0)
@given(key=st_valid_keys, val=st_valid_values)
def test_metadata_key_values(key, val, w_metadata_co):
    w_metadata_co.metadata[key] = val
    assert w_metadata_co.metadata[key] == val