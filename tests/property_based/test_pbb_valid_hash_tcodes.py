import pytest
import string
from hypothesis.strategies import text, integers, one_of
from hypothesis import assume, given, settings


st_text_codes = text(alphabet=string.printable, max_size=2)
st_int_codes = integers(min_value=-99, max_value=99)
st_type_codes = one_of(st_text_codes, st_int_codes)


@given(type_code=st_type_codes)
@settings(max_examples=1000)
def test_verify_commit_ref_typecode(type_code):
    from hangar.records.parsing import cmt_final_digest

    assume(type_code != 'a')
    with pytest.raises(ValueError):
        cmt_final_digest(None, None, None, tcode=type_code)


@given(type_code=st_type_codes)
@settings(max_examples=1000)
def test_verify_array_hash_typecode(type_code):
    from hangar.records.hashmachine import array_hash_digest

    assume(type_code != '0')
    with pytest.raises(ValueError):
        array_hash_digest(None, tcode=type_code)


@given(type_code=st_type_codes)
@settings(max_examples=1000)
def test_verify_schema_hash_typecode(type_code):
    from hangar.records.hashmachine import schema_hash_digest

    assume(type_code != '1')
    with pytest.raises(ValueError):
        schema_hash_digest(None, None, None, None, None, None, None, tcode=type_code)



@given(type_code=st_type_codes)
@settings(max_examples=1000)
def test_verify_metadata_hash_typecode(type_code):
    from hangar.records.hashmachine import metadata_hash_digest

    assume(type_code != '2')
    with pytest.raises(ValueError):
        metadata_hash_digest(None, tcode=type_code)
