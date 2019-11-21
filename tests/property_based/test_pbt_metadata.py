import pytest

import string

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st


st_valid_names = st.text(min_size=1, alphabet=string.ascii_letters + string.digits + '_-.', max_size=16)
st_valid_ints = st.integers(min_value=0, max_value=999)
st_valid_keys = st.one_of(st_valid_ints, st_valid_names)

st_valid_values = st.text(min_size=1, alphabet=string.printable + string.whitespace, max_size=16)


@settings(max_examples=100, deadline=100.0, suppress_health_check=[HealthCheck.too_slow])
@given(key=st_valid_keys, val=st_valid_values)
def test_metadata_key_values(key, val, w_metadata_co):
    w_metadata_co.metadata[key] = val
    assert w_metadata_co.metadata[key] == val