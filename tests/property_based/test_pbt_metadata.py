import pytest

import string

from hypothesis import given, settings
import hypothesis.strategies as st


st_valid_names = st.text(min_size=1, alphabet=string.ascii_letters + string.digits + '_-.', max_size=64)
st_valid_ints = st.integers(min_value=0)
st_valid_keys = st.one_of(st_valid_ints, st_valid_names)

st_valid_values = st.text(min_size=1, alphabet=string.printable)


@settings(max_examples=500)
@given(key=st_valid_keys, val=st_valid_values)
def test_metadata_key_values(key, val, w_checkout):
    w_checkout.metadata[key] = val
    assert w_checkout.metadata[key] == val