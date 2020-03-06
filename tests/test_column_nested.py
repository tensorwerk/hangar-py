"""Tests for the class methods contained in the nested subsample column accessor.
"""
import pytest
import numpy as np
from conftest import fixed_shape_backend_params, variable_shape_backend_params


# --------------------------- Setup ------------------------------


def assert_equal(arr, arr2):
    assert np.array_equal(arr, arr2)
    assert arr.dtype == arr2.dtype


# ------------------------ Tests ----------------------------------


class TestArraysetSetup:

    @pytest.mark.parametrize('name', [
        'invalid\n', '\ninvalid', 'inv name', 'inva@lid', 12, ' try', 'andthis ',
        'VeryLongNameIsInvalidOver64CharactersNotAllowedVeryLongNameIsInva'])
    def test_does_not_allow_invalid_arrayset_names(self, repo, randomsizedarray, name):
        co = repo.checkout(write=True)
        with pytest.raises(ValueError):
            co.add_ndarray_column(name, prototype=randomsizedarray, contains_subsamples=True)
        co.close()

    def test_read_only_mode_arrayset_methods_limited(self, aset_subsamples_initialized_repo):
        import hangar
        co = aset_subsamples_initialized_repo.checkout()
        assert isinstance(co, hangar.checkout.ReaderCheckout)
        with pytest.raises(AttributeError):
            assert co.add_ndarray_column('foo')
        with pytest.raises(AttributeError):
            assert co.add_str_column('foo')
        with pytest.raises(PermissionError):
            assert co.columns.delete('foo')
        assert len(co.columns['writtenaset']) == 0
        co.close()

    def test_get_arrayset_in_read_and_write_checkouts(self, aset_subsamples_initialized_repo, array5by7):
        co = aset_subsamples_initialized_repo.checkout(write=True)
        # getting the column with `get`
        asetOld = co.columns.get('writtenaset')
        asetOldPath = asetOld._path
        asetOldAsetn = asetOld.column
        asetOldDefaultSchemaHash = asetOld._schema.schema_hash_digest()
        co.close()

        co = aset_subsamples_initialized_repo.checkout()
        # getting column with dictionary like style method
        asetNew = co.columns['writtenaset']
        assert asetOldPath == asetNew._path
        assert asetOldAsetn == asetNew.column
        assert asetOldDefaultSchemaHash == asetNew._schema.schema_hash_digest()
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_delete_arrayset(self, aset_backend, aset_subsamples_initialized_repo):
        co = aset_subsamples_initialized_repo.checkout(write=True)
        co.columns.delete('writtenaset')
        assert 'writtenaset' not in co.columns
        with pytest.raises(KeyError):
            # cannot delete twice
            co.columns.delete('writtenaset')

        # init and immediate delete leaves no trace
        co.add_ndarray_column(name='writtenaset', shape=(5, 7), dtype=np.float64,
                                 backend=aset_backend, contains_subsamples=True)
        assert len(co.columns) == 1
        co.columns.delete('writtenaset')
        assert len(co.columns) == 0
        co.commit('this is a commit message')
        co.close()

        # init column in checkout persists aset records/accessor even if no samples contained
        co = aset_subsamples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 0
        co.add_ndarray_column(name='writtenaset', shape=(5, 7), dtype=np.float64,
                                 backend=aset_backend, contains_subsamples=True)
        co.commit('this is a commit message')
        co.close()
        co = aset_subsamples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 1

        # column can be deleted with via __delitem__ dict style command.
        del co.columns['writtenaset']
        assert len(co.columns) == 0
        co.commit('this is a commit message')
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_init_same_arrayset_twice_fails_again(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.add_ndarray_column('aset', prototype=randomsizedarray,
                                 backend=aset_backend, contains_subsamples=True)
        with pytest.raises(LookupError):
            # test if everything is the same as initalized one.
            co.add_ndarray_column('aset', prototype=randomsizedarray,
                                     backend=aset_backend, contains_subsamples=True)
        with pytest.raises(LookupError):
            # test if column container type is different than existing name (no subsamples0
            co.add_ndarray_column('aset', prototype=randomsizedarray,
                                     backend=aset_backend, contains_subsamples=False)
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_arrayset_with_invalid_dimension_sizes_shapes(self, aset_backend, repo):
        co = repo.checkout(write=True)

        shape = (0, 1, 2)
        with pytest.raises(ValueError):
            # cannot have zero valued size for any dimension
            co.add_ndarray_column('aset', shape=shape, dtype=np.int,
                                     backend=aset_backend, contains_subsamples=True)

        shape = [1] * 31
        aset = co.add_ndarray_column('aset1', shape=shape, dtype=np.int,
                                        backend=aset_backend, contains_subsamples=True)
        assert len(aset.shape) == 31

        shape = [1] * 32
        with pytest.raises(ValueError):
            # maximum tensor rank must be <= 31
            co.add_ndarray_column('aset2', shape=shape, dtype=np.int,
                                     backend=aset_backend, contains_subsamples=True)
        co.close()


# ------------------------------ Add Data Tests --------------------------------------------

@pytest.fixture(params=[1, 3], scope='class')
def multi_item_generator(request):
    yield request.param


@pytest.fixture(params=[
    # specifies container types, two-elements: ['outer', 'inner']
    ['dict', None],
    ['list', 'tuple'],
    ['tuple', 'list'],
], scope='class')
def iterable_subsamples(request, multi_item_generator):
    outer, inner = request.param
    arrays = []
    for num_item in range(multi_item_generator):
        arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
        arr += 1
        arrays.append(arr)

    components = []
    for idx, array in enumerate(arrays):
        if inner == 'list':
            component = [f'subsample{idx}', array]
        elif inner == 'tuple':
            component = (f'subsample{idx}', array)
        elif inner is None:
            component = {f'subsample{idx}': array}
        else:
            raise ValueError(
                f'unknown parameter of `inner` {inner} in test suite generation')
        components.append(component)

    if outer == 'dict':
        res = {}
        for part in components:
            res.update(part)
    elif outer == 'list':
        res = []
        for part in components:
            res.append(part)
    elif outer == 'tuple':
        res = []
        for part in components:
            res.append(part)
        res = tuple(res)
    else:
        raise ValueError(
            f'unknown parameter of `outer` {outer} in test suite generation')
    return res


@pytest.fixture(params=['dict', 'list', 'tuple'], scope='class')
def iterable_samples(request, multi_item_generator, iterable_subsamples):
    container = request.param

    if container == 'dict':
        res = {}
        for idx in range(multi_item_generator):
            res[f'sample{idx}'] = iterable_subsamples
    elif container == 'list':
        res = []
        for idx in range(multi_item_generator):
            res.append([f'sample{idx}', iterable_subsamples])
    elif container == 'tuple':
        res = []
        for idx in range(multi_item_generator):
            res.append([f'sample{idx}', iterable_subsamples])
        res = tuple(res)
    else:
        raise ValueError(
            f'unknown parameter of `container` {container} in test suite generation')
    return res


@pytest.fixture(params=fixed_shape_backend_params, scope='class')
def backend_params(request):
    return request.param


@pytest.fixture()
def subsample_writer_written_aset(backend_params, repo, monkeypatch):
    from hangar.backends import hdf5_00
    from hangar.backends import hdf5_01
    from hangar.backends import numpy_10
    monkeypatch.setattr(hdf5_00, 'COLLECTION_COUNT', 5)
    monkeypatch.setattr(hdf5_00, 'COLLECTION_SIZE', 10)
    monkeypatch.setattr(hdf5_01, 'COLLECTION_COUNT', 5)
    monkeypatch.setattr(hdf5_01, 'COLLECTION_SIZE', 10)
    monkeypatch.setattr(numpy_10, 'COLLECTION_SIZE', 10)

    co = repo.checkout(write=True)
    aset = co.add_ndarray_column('foo', shape=(4, 4), dtype=np.uint8, variable_shape=False,
                                    backend=backend_params, contains_subsamples=True)
    yield aset
    co.close()


class TestAddData:

    def test_update_sample_subsamples_empty_arrayset(self, subsample_writer_written_aset, iterable_samples):
        aset = subsample_writer_written_aset
        added = aset.update(iterable_samples)
        assert added is None
        assert len(aset._samples) == len(iterable_samples)
        for sample_idx, sample_data in enumerate(iterable_samples):
            assert f'sample{sample_idx}' in aset._samples

    def test_update_sample_kwargs_only_empty_arrayset(self, subsample_writer_written_aset, iterable_subsamples):
        aset = subsample_writer_written_aset
        added = aset.update(fookwarg=iterable_subsamples)
        assert added is None
        assert len(aset._samples) == 1
        assert 'fookwarg' in aset._samples

        added = aset.update(bar=iterable_subsamples, baz=iterable_subsamples)
        assert added is None
        assert len(aset._samples) == 3
        assert 'bar' in aset._samples
        assert 'baz' in aset._samples
        for subsample_idx, _data in enumerate(iterable_subsamples):
            assert f'subsample{subsample_idx}' in aset._samples['fookwarg']._subsamples
            assert f'subsample{subsample_idx}' in aset._samples['bar']._subsamples
            assert f'subsample{subsample_idx}' in aset._samples['baz']._subsamples

    def test_update_sample_kwargs_and_other_dict_doesnt_modify_input_in_calling_scope(
        self, subsample_writer_written_aset, iterable_subsamples, iterable_samples
    ):
        """ensure bug does not revert.

        Had a case where if dict was passed as ``other`` along with kwargs, the operation
        would complete as normally, but when control returned to the caller the original
        dict passed in as ``other`` would have been silently merged with the kwargs.
        """
        aset = subsample_writer_written_aset
        if not isinstance(iterable_samples, dict):
            return
        iterable_samples_before = list(iterable_samples.items())

        aset.update(iterable_samples, kwargadded=iterable_subsamples)
        # in bug case, would now observe that iterable_samples would have been
        # silently modified in a method analogous to calling:
        #
        #   ``iterable_samples.update({'kwargadded': iterable_subsamples})``
        #
        assert list(iterable_samples.items()) == iterable_samples_before

    def test_update_sample_kwargs_and_iterably_empty_arrayset(self, subsample_writer_written_aset, iterable_subsamples, iterable_samples):
        aset = subsample_writer_written_aset
        aset.update(iterable_samples, fookwarg=iterable_subsamples)
        assert len(aset._samples) == len(iterable_samples) + 1

        assert 'fookwarg' in aset._samples
        for sample_idx in range(len(iterable_samples)):
            assert f'sample{sample_idx}' in aset._samples

    def test_update_sample_subsamples_duplicate_data_does_not_save_new(self, subsample_writer_written_aset, iterable_samples):
        aset = subsample_writer_written_aset
        aset.update(iterable_samples)
        old_specs = {}
        for sample_idx, sample_data in enumerate(iterable_samples):
            old_specs[f'sample{sample_idx}'] = aset._samples[f'sample{sample_idx}']._subsamples.copy()

        aset.update(iterable_samples)
        new_specs = {}
        for sample_idx, sample_data in enumerate(iterable_samples):
            new_specs[f'sample{sample_idx}'] = aset._samples[f'sample{sample_idx}']._subsamples.copy()
        assert old_specs == new_specs

    def test_update_sample_subsamples_context_manager(self, subsample_writer_written_aset, iterable_samples):
        aset = subsample_writer_written_aset
        assert aset._is_conman is False
        with aset as cm_aset:
            assert cm_aset._is_conman is True
            added = cm_aset.update(iterable_samples)
            assert added is None
        assert aset._is_conman is False

        assert len(aset._samples) == len(iterable_samples)
        for sample_idx, sample_data in enumerate(iterable_samples):
            assert f'sample{sample_idx}' in aset._samples

    def test_setitem_sample_subsamples_empty_arrayset(
            self, multi_item_generator, subsample_writer_written_aset, iterable_subsamples
    ):
        aset = subsample_writer_written_aset

        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = iterable_subsamples
        assert len(aset._samples) == len(iterable_subsamples)

        for sample_idx in range(multi_item_generator):
            assert f'sample{sample_idx}' in aset._samples
            assert len(aset._samples[f'sample{sample_idx}']._subsamples) == len(iterable_subsamples)
            for subsample_idx in range(len(iterable_subsamples)):
                assert f'subsample{subsample_idx}' in aset._samples[f'sample{sample_idx}']._subsamples

    def test_setitem_sample_subsamples_contextmanager(
            self, multi_item_generator, subsample_writer_written_aset, iterable_subsamples
    ):
        aset = subsample_writer_written_aset
        assert aset._is_conman is False
        with aset as aset_cm:
            assert aset_cm._is_conman is True
            for sample_idx in range(multi_item_generator):
                aset_cm[f'sample{sample_idx}'] = iterable_subsamples
            assert len(aset_cm._samples) == len(iterable_subsamples)
            assert aset_cm._samples[f'sample{sample_idx}']._is_conman is True
        assert aset._is_conman is False

        for sample_idx in range(multi_item_generator):
            assert f'sample{sample_idx}' in aset._samples
            assert len(aset._samples[f'sample{sample_idx}']._subsamples) == len(iterable_subsamples)
            for subsample_idx in range(len(iterable_subsamples)):
                assert f'subsample{subsample_idx}' in aset._samples[
                    f'sample{sample_idx}']._subsamples

    def test_update_subsamples_empty_arrayset(self, multi_item_generator, subsample_writer_written_aset, iterable_subsamples):
        aset = subsample_writer_written_aset
        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + 10}
            aset[f'sample{sample_idx}'].update(iterable_subsamples)
        assert len(aset._samples) == len(iterable_subsamples)

        for sample_idx in range(multi_item_generator):
            assert f'sample{sample_idx}' in aset._samples
            assert len(aset._samples[f'sample{sample_idx}']._subsamples) == len(iterable_subsamples) + 1
            assert 'foo' in aset._samples[f'sample{sample_idx}']._subsamples
            for subsample_idx in range(len(iterable_subsamples)):
                assert f'subsample{subsample_idx}' in aset._samples[f'sample{sample_idx}']._subsamples

    def test_update_subsamples_via_kwargs_empty_arrayset(self, multi_item_generator, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + 10}
            aset[f'sample{sample_idx}'].update(bar=np.arange(16, dtype=np.uint8).reshape(4, 4) + 20)
        assert len(aset._samples) == multi_item_generator

        for sample_idx in range(multi_item_generator):
            assert f'sample{sample_idx}' in aset._samples
            assert len(aset._samples[f'sample{sample_idx}']._subsamples) == 2
            assert 'foo' in aset._samples[f'sample{sample_idx}']._subsamples
            assert 'bar' in aset._samples[f'sample{sample_idx}']._subsamples

    def test_update_subsamples_kwargs_and_other_dict_doesnt_modify_input_in_calling_scopy(
        self, multi_item_generator, subsample_writer_written_aset, iterable_subsamples
    ):
        """ensure bug does not revert.

        Had a case where if dict was passed as ``other`` along with kwargs, the operation
        would complete as normally, but when control returned to the caller the original
        dict passed in as ``other`` would have been silently merged with the kwargs.
        """
        aset = subsample_writer_written_aset
        if not isinstance(iterable_subsamples, dict):
            return
        iterable_subsamples_before = list(iterable_subsamples.keys())

        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + 10}
            aset[f'sample{sample_idx}'].update(iterable_subsamples, kwargadded=np.arange(16, dtype=np.uint8).reshape(4, 4))
            # in bug case, would now observe that iterable_subsamples would have been
            # silently modified in a method analogous to calling:
            #
            #   ``iterable_subsamples.update({'kwargadded': np.array})``
            #
            assert list(iterable_subsamples.keys()) == iterable_subsamples_before
        assert list(iterable_subsamples.keys()) == iterable_subsamples_before

    def test_update_subsamples_via_kwargs_and_iterable_empty_arrayset(
        self, multi_item_generator, subsample_writer_written_aset, iterable_subsamples
    ):
        aset = subsample_writer_written_aset
        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + 10}
            aset[f'sample{sample_idx}'].update(iterable_subsamples, bar=np.arange(16, dtype=np.uint8).reshape(4, 4))

        assert len(aset._samples) == multi_item_generator

        for sample_idx in range(multi_item_generator):
            assert f'sample{sample_idx}' in aset._samples
            assert len(aset._samples[f'sample{sample_idx}']._subsamples) == len(iterable_subsamples) + 2
            assert 'foo' in aset._samples[f'sample{sample_idx}']._subsamples
            assert 'bar' in aset._samples[f'sample{sample_idx}']._subsamples

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_update_subsamples_context_manager(self, backend, multi_item_generator,
                                               iterable_subsamples, repo):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('foo', shape=(4, 4), dtype=np.uint8,
                                        backend=backend, contains_subsamples=True)

        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + 10}
            assert aset._is_conman is False
            with aset[f'sample{sample_idx}'] as sample_cm:
                assert sample_cm._is_conman is True
                assert aset._is_conman is True
                sample_cm.update(iterable_subsamples)
            assert aset._is_conman is False
        assert len(aset._samples) == len(iterable_subsamples)

        for sample_idx in range(multi_item_generator):
            assert f'sample{sample_idx}' in aset._samples
            assert len(aset._samples[f'sample{sample_idx}']._subsamples) == len(iterable_subsamples) + 1
            assert 'foo' in aset._samples[f'sample{sample_idx}']._subsamples
            for subsample_idx in range(len(iterable_subsamples)):
                assert f'subsample{subsample_idx}' in aset._samples[f'sample{sample_idx}']._subsamples
        co.close()

    def test_setitem_sample_empty_arrayset(self, multi_item_generator, iterable_subsamples, subsample_writer_written_aset):
        aset = subsample_writer_written_aset

        subsamples_dict = dict(iterable_subsamples)
        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + 10}
            for subsample_key, subsample_val in subsamples_dict.items():
                aset[f'sample{sample_idx}'][subsample_key] = subsample_val
        assert len(aset._samples) == len(iterable_subsamples)

        for sample_idx in range(multi_item_generator):
            assert f'sample{sample_idx}' in aset._samples
            assert len(aset._samples[f'sample{sample_idx}']._subsamples) == len(subsamples_dict) + 1
            assert 'foo' in aset._samples[f'sample{sample_idx}']._subsamples
            for subkey in subsamples_dict.keys():
                assert subkey in aset._samples[f'sample{sample_idx}']._subsamples

    def test_setitem_sample_setitem_subsample_empty_arrayset_fails(self, subsample_writer_written_aset):
        """This should fail because __getitem___ raises keyerror when

        ``aset[foo-sample][subsample] = np.ndarray`` runs.

        The ``aset[foo-sample]`` part fails with KeyError, and no subsample
        accessor is returned for the __setitem__ call following __getitem__
        """
        aset = subsample_writer_written_aset
        with pytest.raises(KeyError, match='sample'):
            aset['sample']
        with pytest.raises(KeyError, match='sample'):
            aset['sample']['subsample'] = np.arange(16, dtype=np.uint8).reshape(4, 4)
        assert len(aset) == 0

    def test_setitem_subsamples_contextmanager(self, multi_item_generator, iterable_subsamples, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        subsamples_dict = dict(iterable_subsamples)
        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + 10}
            assert aset._is_conman is False
            with aset[f'sample{sample_idx}'] as sample_cm:
                assert sample_cm._is_conman is True
                assert aset._is_conman is True
                for subsample_key, subsample_val in subsamples_dict.items():
                    sample_cm[subsample_key] = subsample_val
            assert aset._is_conman is False
        assert len(aset._samples) == len(iterable_subsamples)

        for sample_idx in range(multi_item_generator):
            assert f'sample{sample_idx}' in aset._samples
            assert len(aset._samples[f'sample{sample_idx}']._subsamples) == len(subsamples_dict) + 1
            assert 'foo' in aset._samples[f'sample{sample_idx}']._subsamples
            for subkey in subsamples_dict.keys():
                assert subkey in aset._samples[f'sample{sample_idx}']._subsamples

    def test_append_subsamples_empty_arrayset(self, multi_item_generator, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {
                'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + ((sample_idx * 2) + 1)
            }
            outkey = aset[f'sample{sample_idx}'].append(
                np.arange(16, dtype=np.uint8).reshape(4, 4) + sample_idx
            )
            assert 'foo' in aset._samples[f'sample{sample_idx}']._subsamples
            assert outkey in aset._samples[f'sample{sample_idx}']._subsamples
        assert len(aset._samples) == multi_item_generator

    def test_append_subsamples_contextmanager(self, multi_item_generator, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        for sample_idx in range(multi_item_generator):
            aset[f'sample{sample_idx}'] = {
                'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + ((sample_idx * 2) + 1)
            }
            assert aset._is_conman is False
            with aset[f'sample{sample_idx}'] as sample_cm:
                assert aset._is_conman is True
                assert sample_cm._is_conman is True
                outkey = sample_cm.append(np.arange(16, dtype=np.uint8).reshape(4, 4) + sample_idx)
            assert aset._is_conman is False
            assert 'foo' in aset._samples[f'sample{sample_idx}']._subsamples
            assert outkey in aset._samples[f'sample{sample_idx}']._subsamples
        assert len(aset._samples) == multi_item_generator

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    @pytest.mark.parametrize('other', [
        [f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)],
        (f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)),
    ])
    def test_update_noniterable_subsample_iter_fails(self, backend, other, repo):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('foo', shape=(4, 4), dtype=np.uint8,
                                        backend=backend, contains_subsamples=True)
        aset[f'foo'] = {'foo': np.arange(16, dtype=np.uint8).reshape(4, 4) + 10}
        with pytest.raises(ValueError, match='dictionary update sequence'):
            aset['foo'].update(other)
        assert len(aset._samples) == 1
        assert len(aset._samples['foo']._subsamples) == 1
        assert 'foo' in aset._samples['foo']._subsamples
        assert 'subsample1' not in aset._samples['foo']._subsamples
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_update_subsamples_with_too_many_arguments_fails(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('foo', shape=(4, 4), dtype=np.uint8,
                                        backend=backend, contains_subsamples=True)
        arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
        aset[f'foo'] = {'foo': arr + 10}
        with pytest.raises(TypeError, match='takes from 1 to 2 positional arguments'):
            aset['foo'].update('fail', arr)
        assert len(aset._samples) == 1
        assert len(aset._samples['foo']._subsamples) == 1
        assert 'foo' in aset._samples['foo']._subsamples
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_update_subsamples_with_too_few_arguments_fails(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('foo', shape=(4, 4), dtype=np.uint8,
                                        backend=backend, contains_subsamples=True)
        arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
        aset[f'foo'] = {'foo': arr + 10}
        with pytest.raises(ValueError, match='dictionary update sequence element #0 has length 1; 2 is required'):
            aset['foo'].update('fail')
        assert len(aset._samples) == 1
        assert len(aset._samples['foo']._subsamples) == 1
        assert 'foo' in aset._samples['foo']._subsamples
        co.close()

    @pytest.mark.parametrize('other', [
        ['sample1', [[f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)]]],
        ['sample1', ((f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)),)],
        ('sample1', ((f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)),),),
        ('sample1', [[f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)]],),
        ['sample1', ([f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)])],
        ['sample1', [(f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4))]],
        ('sample1', ([f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)]),),
        ('sample1', [(f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4))],),
        ['sample1', [f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)]],
        ['sample1', (f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4))],
        ('sample1', [f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)],),
        ('sample1', (f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)),),
        ('sample1', {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)},),
        ['sample1', {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}],
    ])
    def test_update_noniterable_samples_fails(self, other, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        with pytest.raises(ValueError, match='dictionary update sequence'):
            aset.update(other)
        assert len(aset._samples) == 0

    @pytest.mark.parametrize('other', [
        [['sample1', [f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)]]],
        [['sample1', (f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4),)]],
        (('sample1', (f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4))),),
        (('sample1', [f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)]),),
        {'sample1': [f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)]},
        {'sample1': (f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4))},
        {'sample1': (f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4))},
        {'sample1': [f'subsample1', np.arange(16, dtype=np.uint8).reshape(4, 4)]},
    ])
    def test_update_noniterable_subsamples_fails(self, other, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        with pytest.raises(ValueError, match='dictionary update sequence'):
            aset.update(other)
        assert len(aset._samples) == 0

    @pytest.mark.parametrize('other', [
        {'sample1!': {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {-2: {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'lol cat': {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'sample 1': {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {('sample', 'one'): {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {(1, 2): {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {('sample', 2): {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {(1, 'sample'): {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
    ])
    def test_update_invalid_sample_key_fails(self, other, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        with pytest.raises(ValueError, match='is not suitable'):
            aset.update(other)
        assert len(aset._samples) == 0

    @pytest.mark.parametrize('other', [
        {'sample': {f'subsample1!': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'sample': {f'subsample 1': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'sample': {-2: np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'sample': {f'subsample1\n': np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'sample': {(1, 2): np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'sample': {('s1', 's2'): np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'sample': {('s1', 1): np.arange(16, dtype=np.uint8).reshape(4, 4)}},
        {'sample': {(1, 's1'): np.arange(16, dtype=np.uint8).reshape(4, 4)}},
    ])
    def test_update_sample_invalid_subsample_key_fails(self, other, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        with pytest.raises(ValueError, match='is not suitable'):
            aset.update(other)
        assert len(aset._samples) == 0

    @pytest.mark.parametrize('variable_shape,backend', [
        *[[True, be] for be in variable_shape_backend_params],
        *[[False, be] for be in fixed_shape_backend_params],
    ])
    @pytest.mark.parametrize('other', [
        {'sample': {f'subsample1': np.arange(9, dtype=np.uint8).reshape(3, 3)}},
        {'sample': {f'subsample1': np.arange(8, dtype=np.uint8).reshape(2, 2, 2)}},
        {'sample': {f'subsample1': np.arange(4, dtype=np.float32).reshape(2, 2)}},
        {'sample': {f'subsample1': np.arange(4, dtype=np.uint8).reshape((2, 2), order='F')}},
        {'sample': {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4).tolist()}},
    ])
    def test_update_sample_invalid_array_fails_fixed_shape(self, backend, variable_shape, other, repo):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('foo',
                                        shape=(2, 2), dtype=np.uint8, variable_shape=variable_shape,
                                        backend=backend, contains_subsamples=True)
        with pytest.raises(ValueError):
            aset.update(other)
        assert len(aset._samples) == 0
        co.close()

    @pytest.mark.parametrize('other', [
        {f'subsample1!': np.arange(16, dtype=np.uint8).reshape(4, 4)},
        {f'subsample 1': np.arange(16, dtype=np.uint8).reshape(4, 4)},
        {-2: np.arange(16, dtype=np.uint8).reshape(4, 4)},
        {f'subsample1\n': np.arange(16, dtype=np.uint8).reshape(4, 4)},
        {(1, 2): np.arange(16, dtype=np.uint8).reshape(4, 4)},
        {('s1', 's2'): np.arange(16, dtype=np.uint8).reshape(4, 4)},
        {('s1', 1): np.arange(16, dtype=np.uint8).reshape(4, 4)},
        {(1, 's1'): np.arange(16, dtype=np.uint8).reshape(4, 4)},
    ])
    def test_update_subsample_invalid_subsample_key_fails(self, other, subsample_writer_written_aset):
        aset = subsample_writer_written_aset
        aset['sample'] = {0: np.zeros((4, 4), dtype=np.uint8)}
        with pytest.raises(ValueError, match='is not suitable'):
            aset['sample'].update(other)
        assert len(aset._samples) == 1
        assert len(aset._samples['sample']._subsamples) == 1
        assert 0 in aset._samples['sample']._subsamples

    @pytest.mark.parametrize('variable_shape,backend', [
        *[[False, be] for be in fixed_shape_backend_params],
    ])
    @pytest.mark.parametrize('other', [
        {f'subsample1': np.arange(9, dtype=np.uint8).reshape(3, 3)},
        {f'subsample1': np.arange(8, dtype=np.uint8).reshape(2, 2, 2)},
        {f'subsample1': np.arange(4, dtype=np.float32).reshape(2, 2)},
        {f'subsample1': np.arange(4, dtype=np.uint8).reshape((2, 2), order='F')},
        {f'subsample1': np.arange(16, dtype=np.uint8).reshape(4, 4).tolist()},
    ])
    def test_update_subsample_invalid_array_fails_fixed_shape(self, backend, variable_shape, other, repo):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('foo',
                                        shape=(4, 4), dtype=np.uint8, variable_shape=variable_shape,
                                        backend=backend, contains_subsamples=True)
        aset['sample'] = {0: np.zeros((4, 4), dtype=np.uint8)}
        with pytest.raises(ValueError):
            aset['sample'].update(other)
        assert len(aset._samples) == 1
        assert len(aset._samples['sample']._subsamples) == 1
        assert 0 in aset._samples['sample']._subsamples
        co.close()


# --------------------------- Test Remove Data -------------------------------------


@pytest.fixture(scope='class')
def subsample_data_map():
    arr = np.arange(5*7).astype(np.uint16).reshape((5, 7))
    res = {
        'foo': {
            0: arr,
            1: arr + 1,
            2: arr + 2
        },
        2: {
            'bar': arr + 3,
            'baz': arr + 4
        }
    }
    return res


@pytest.fixture(params=fixed_shape_backend_params, scope='class')
def backend_param(request):
    return request.param


@pytest.fixture(params=[False, True], scope='class')
def write_enabled(request):
    return request.param


@pytest.fixture(scope='class')
def initialized_arrayset(write_enabled, backend_param, classrepo, subsample_data_map):
    co = classrepo.checkout(write=True)
    aset = co.add_ndarray_column(f'foo{backend_param}{int(write_enabled)}',
                                    shape=(5, 7), dtype=np.uint16, backend=backend_param,
                                    contains_subsamples=True)
    aset.update(subsample_data_map)
    co.commit(f'done {backend_param}{write_enabled}')
    co.close()
    if write_enabled:
        nco = classrepo.checkout(write=True)
        yield nco.columns[f'foo{backend_param}{int(write_enabled)}']
        nco.close()
    else:
        nco = classrepo.checkout()
        yield nco.columns[f'foo{backend_param}{int(write_enabled)}']
        nco.close()


@pytest.fixture()
def initialized_arrayset_write_only(backend_param, repo, subsample_data_map):
    co = repo.checkout(write=True)
    aset = co.add_ndarray_column('foo', shape=(5, 7), dtype=np.uint16,
                                    backend=backend_param, contains_subsamples=True)
    aset.update(subsample_data_map)
    yield co.columns['foo']
    co.close()


class TestRemoveData:

    # --------------------- delete -----------------------------

    def test_delitem_single_sample_from_arrayset(self, initialized_arrayset_write_only):
        aset = initialized_arrayset_write_only
        del aset['foo']
        assert 'foo' not in aset._samples
        assert 'foo' not in aset

    def test_delitem_single_subsample_from_sample(self, initialized_arrayset_write_only):
        aset = initialized_arrayset_write_only
        del aset['foo'][0]
        assert 0 not in aset._samples['foo']._subsamples
        assert 0 not in aset['foo']

    def test_delitem_sample_nonexisting_keys_fails(self, initialized_arrayset_write_only):
        aset = initialized_arrayset_write_only
        assert 'doesnotexist' not in aset._samples
        assert 'doesnotexist' not in aset
        with pytest.raises(KeyError):
            del aset['doesnotexist']

    def test_delitem_single_subsample_nonexisting_key_fails(self, initialized_arrayset_write_only):
        aset = initialized_arrayset_write_only
        assert 'foo' in aset._samples
        assert 'foo' in aset
        assert 'doesnotexist' not in aset._samples['foo']._subsamples
        assert 'doesnotexist' not in aset['foo']
        with pytest.raises(KeyError):
            del aset['foo']['doesnotexist']

    def test_delitem_multiple_samples_fails_keyerror(self, initialized_arrayset_write_only):
        aset = initialized_arrayset_write_only
        with pytest.raises(KeyError, match="('foo', 2)"):
            del aset['foo', 2]
        assert 'foo' in aset
        assert 2 in aset

    # ------------------------ pop ----------------------------

    def test_pop_single_sample_from_arrayset(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset.pop('foo')
        assert 'foo' not in aset
        assert isinstance(res, dict)
        assert len(res) == len(subsample_data_map['foo'])
        for expected_k, expected_v in subsample_data_map['foo'].items():
            assert_equal(res[expected_k], expected_v)

    def test_pop_multiple_samples_from_arrayset_fails(self, initialized_arrayset_write_only):
        aset = initialized_arrayset_write_only
        with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were"):
            aset.pop('foo', 2)
        assert 'foo' in aset
        assert 2 in aset

    def test_pop_single_subsample_from_sample(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset['foo'].pop(0)
        assert 0 not in aset['foo']
        assert isinstance(res, np.ndarray)
        assert_equal(res, subsample_data_map['foo'][0])

    def test_pop_multiple_subsample_from_sample_fails(self, initialized_arrayset_write_only):
        aset = initialized_arrayset_write_only
        with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
            aset['foo'].pop(*[0, 1])
        assert 0 in aset['foo']
        assert 1 in aset['foo']


# ------------------------------ Container Introspection -----------------------------------


class TestContainerIntrospection:

    def test_get_sample_returns_object(self, initialized_arrayset, subsample_data_map):
        from hangar.columns.layout_nested import FlatSubsampleReader, NestedSampleReader

        aset = initialized_arrayset
        assert isinstance(aset, NestedSampleReader)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample, FlatSubsampleReader)

    # -------------------------- test __dunder__ methods ----------------------------------

    def test_get_sample_test_subsample_len_method(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert len(sample) == len(subsample_data)

    def test_get_sample_test_subsample_contains_method(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name in subsample_data.keys():
                assert subsample_name in sample

    def test_sample_len_reported_correctly(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        assert len(aset) == len(subsample_data_map)
        assert aset.num_subsamples == sum([len(subsample) for subsample in subsample_data_map.values()])

    # ----------------------------- test property ---------------------------

    def test_get_sample_test_subsample_sample_property(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.sample == sample_name

    def test_get_sample_test_subsample_arrayset_property(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.column.startswith('foo')

    def test_get_sample_test_data_property(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample.data
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_test_subsample_contains_remote_references_property(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        # test works before add remote references
        aset.contains_remote_references is False
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.contains_remote_references is False

        # add subsamples which are not local to each subsample
        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        aset['foo']._subsamples[50] = template
        aset[2]._subsamples[50] = template

        aset.contains_remote_references is True
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.contains_remote_references is True

        del aset._samples['foo']._subsamples[50]
        del aset._samples[2]._subsamples[50]

    def test_get_sample_test_subsample_remote_reference_keys_property(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        # test works before add remote references
        assert aset.remote_reference_keys == ()
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.remote_reference_keys == ()

        # add subsamples which are not local to each subsample
        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        aset['foo']._subsamples[50] = template
        aset[2]._subsamples[50] = template

        assert aset.remote_reference_keys == (2, 'foo') or ('foo', 2)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.remote_reference_keys == (50,)

        del aset._samples['foo']._subsamples[50]
        del aset._samples[2]._subsamples[50]

    def test_getattr_does_not_raise_permission_error_if_alive(self, initialized_arrayset):
        aset = initialized_arrayset

        assert hasattr(aset, 'doesnotexist') is False  # does not raise error
        assert hasattr(aset, '_mode') is True
        with pytest.raises(AttributeError):
            assert getattr(aset, 'doesnotexist')
        assert getattr(aset, '_mode') == 'a' if aset.iswriteable else 'r'

        sample = aset['foo']
        assert hasattr(sample, 'doesnotexist') is False  # does not raise error
        assert hasattr(sample, '_mode') is True
        with pytest.raises(AttributeError):
            assert getattr(sample, 'doesnotexist')
        assert getattr(sample, '_mode') == 'a' if aset.iswriteable else 'r'

        # mock up destruct call in sample and aset.
        original = getattr(aset, '_mode')
        delattr(aset, '_mode')
        delattr(sample, '_mode')
        with pytest.raises(PermissionError):
            hasattr(aset, 'doesnotexist')
        with pytest.raises(PermissionError):
            hasattr(aset, '_mode')

        with pytest.raises(PermissionError):
            hasattr(sample, 'doesnotexist')
        with pytest.raises(PermissionError):
            hasattr(sample, '_mode')
        setattr(aset, '_mode', original)
        setattr(sample, '_mode', original)

# ------------------------------ Getting Data --------------------------------------------


class TestGetDataMethods:

    def test_get_sample_missing_key(self, initialized_arrayset):
        aset = initialized_arrayset
        returned = aset.get('doesnotexist')
        assert returned is None
        default_returned = aset.get(9999, default=True)
        assert default_returned is True

    def test_getitem_sample_missing_key(self, initialized_arrayset):
        aset = initialized_arrayset
        with pytest.raises(KeyError):
            aset['doesnotexist']

    def test_get_sample_get_subsample(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name, subsample_value in subsample_data.items():
                res = sample.get(subsample_name)
                assert_equal(res, subsample_value)

    def test_getitem_sample_getitem_subsample(self, initialized_arrayset, subsample_data_map):
        from hangar.columns.layout_nested import FlatSubsampleReader

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset[sample_name]
            assert isinstance(sample, FlatSubsampleReader)
            for subsample_name, subsample_value in subsample_data.items():
                res = sample[subsample_name]
                assert_equal(res, subsample_value)

    def test_get_sample_get_subsample_missing_key(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name in subsample_data_map.keys():
            sample = aset.get(sample_name)
            returned = sample.get('doesnotexist')
            assert returned is None
            default_returned = sample.get(9999, default=True)
            assert default_returned is True

    def test_getitem_sample_getitem_subsample_missing_key(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name in subsample_data_map.keys():
            sample = aset[sample_name]
            with pytest.raises(KeyError):
                sample['doesnotexist']

    def test_get_sample_get_multiple_subsamples_fails(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            with pytest.raises(TypeError):
                sample.get(*list(list(subsample_data.keys())[:2]), default=None)

    def test_get_sample_getitem_single_subsample(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name, subsample_value in subsample_data.items():
                res = sample[subsample_name]
                assert_equal(res, subsample_value)

    def test_get_sample_getitem_single_subsample_missing_key(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name in subsample_data_map.keys():
            sample = aset.get(sample_name)
            returned = sample.get('doesnotexist')
            assert returned is None
            default_returned = sample.get(9999, default=True)
            assert default_returned is True

    def test_get_sample_getitem_multiple_subsamples_fails(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            with pytest.raises(TypeError):
                sample[list(subsample_data.keys())[:2]]

    def test_get_sample_getitem_subsamples_with_ellipsis(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[...]
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_getitem_subsamples_with_keys_and_ellipsis_fails(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            existing_subsample_key = next(iter(subsample_data.keys()))
            with pytest.raises(TypeError):
                sample[..., existing_subsample_key]
            with pytest.raises(TypeError):
                sample[..., [existing_subsample_key]]

    def test_get_sample_getitem_subsamples_with_unbound_slice(self, initialized_arrayset, subsample_data_map):
        """unbound slice is ``slice(None) == [:]``"""
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[:]
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_getitem_subsamples_with_bounded_slice(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[0:2]
            assert isinstance(res, dict)
            assert len(res) == 2
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_getitem_subsamples_with_out_of_bounds_slice_does_not_fail(
            self, initialized_arrayset, subsample_data_map):
        """Odd python behavior we emulate: out of bounds sequence slicing is allowed.

        Instead of throwing an exception, the slice is treated as if it should just
        go up to the total number of elements in the container. For example:
            [1, 2, 3][0:5] == [1, 2, 3]
        """
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[0:5]
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_aset_contextmanager(self, initialized_arrayset, subsample_data_map):
        assert initialized_arrayset._is_conman is False
        with initialized_arrayset as aset:
            assert aset._is_conman is True
            for sample_name, subsample_data in subsample_data_map.items():
                sample = aset.get(sample_name)
                assert sample._is_conman is True
                for subsample_name, expected_val in subsample_data.items():
                    assert_equal(sample.get(subsample_name), expected_val)
                assert sample._is_conman is True
        assert initialized_arrayset._is_conman is False
        assert aset._is_conman is False
        assert sample._is_conman is False

    def test_sample_contextmanager(self, initialized_arrayset, subsample_data_map):
        for sample_name, subsample_data in subsample_data_map.items():
            sample = initialized_arrayset.get(sample_name)
            assert initialized_arrayset._is_conman is False
            assert sample._is_conman is False
            with sample as sample_cm:
                assert sample_cm._is_conman is True
                assert initialized_arrayset._is_conman is True
                for subsample_name, expected_val in subsample_data.items():
                    assert_equal(sample_cm.get(subsample_name), expected_val)
            assert sample._is_conman is False
            assert initialized_arrayset._is_conman is False
        assert initialized_arrayset._is_conman is False
        assert sample._is_conman is False

    def test_sample_subsample_contextmanager(self, initialized_arrayset, subsample_data_map):
        assert initialized_arrayset._is_conman is False
        with initialized_arrayset as aset:
            assert aset._is_conman is True
            assert aset._enter_count == 1
            for sample_name, subsample_data in subsample_data_map.items():
                sample = aset.get(sample_name)
                assert sample._is_conman is True
                assert sample._enter_count == 1
                with sample as sample_cm:
                    assert aset._is_conman is True
                    assert sample_cm._is_conman is True
                    assert aset._enter_count == 2
                    assert sample_cm._enter_count == 2
                    for subsample_name, expected_val in subsample_data.items():
                        assert_equal(sample_cm.get(subsample_name), expected_val)
                assert aset._is_conman is True
                assert sample_cm._is_conman is True
                assert aset._enter_count == 1
                assert sample_cm._enter_count == 1
        assert initialized_arrayset._is_conman is False
        assert aset._is_conman is False
        assert sample._is_conman is False
        assert aset._enter_count == 0
        assert sample_cm._enter_count == 0

    def test_sample_reentrant_contextmanager_fails(self, initialized_arrayset, subsample_data_map):
        assert initialized_arrayset._is_conman is False

        with initialized_arrayset as aset:
            assert aset._is_conman is True
            assert aset._enter_count == 1
            for sample_name, subsample_data in subsample_data_map.items():
                sample = aset.get(sample_name)
                assert sample._is_conman is True
                assert sample._enter_count == 1
                with sample as sample_cm:
                    assert aset._is_conman is True
                    assert sample_cm._is_conman is True
                    assert aset._enter_count == 2
                    assert sample_cm._enter_count == 2
                    for subsample_name, expected_val in subsample_data.items():
                        assert_equal(sample_cm.get(subsample_name), expected_val)
                # reentrant demonstrated here here
                with sample as sample_cm2:
                    assert aset._is_conman is True
                    assert sample_cm._is_conman is True
                    assert sample_cm2._is_conman is True
                    assert aset._enter_count == 2
                    assert sample_cm._enter_count == 2
                    assert sample_cm2._enter_count == 2
                    for subsample_name, expected_val in subsample_data.items():
                        assert_equal(sample_cm2.get(subsample_name), expected_val)
                assert aset._is_conman is True
                assert sample_cm._is_conman is True
                assert aset._enter_count == 1
                assert sample_cm._enter_count == 1
        assert initialized_arrayset._is_conman is False
        assert aset._is_conman is False
        assert sample._is_conman is False
        assert aset._enter_count == 0
        assert sample_cm._enter_count == 0

    # -------------------------- dict-style iteration methods ---------------------------

    def test_calling_iter_on_arrayset(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        arrayset_it = iter(aset)  # returns iterator over sample keys
        for sample_name in arrayset_it:
            assert sample_name in aset
            assert sample_name in subsample_data_map

    def test_calling_iter_on_sample_in_arrayset(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        arrayset_it = iter(aset)  # returns iterator over sample keys
        for sample_name in arrayset_it:
            assert sample_name in aset
            assert sample_name in subsample_data_map

            sample_it = iter(aset[sample_name])  # returns iterator over subsample keys
            for subsample_name in sample_it:
                assert subsample_name in aset[sample_name]
                assert subsample_name in subsample_data_map[sample_name]

    def test_get_sample_keys_method(self, initialized_arrayset):
        from collections.abc import Iterator
        aset = initialized_arrayset

        assert isinstance(aset.keys(), Iterator)
        res = list(aset.keys())
        assert len(res) == 2
        assert 2 and 'foo' in res

    def test_get_sample_keys_method_local_only(self, initialized_arrayset):
        from collections.abc import Iterator
        aset = initialized_arrayset

        # add subsamples which are not local to each subsample
        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        aset['foo']._subsamples[50] = template

        assert isinstance(aset.keys(local=True), Iterator)
        res = list(aset.keys(local=True))
        assert len(res) == 1
        assert 2 in res

        del aset._samples['foo']._subsamples[50]

    def test_get_sample_subsample_keys_method(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.keys(), Iterator)
            res = list(sample.keys())
            for k in res:
                assert k in subsample_data

    def test_get_sample_subsample_keys_method_local_only(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator
        aset = initialized_arrayset

        # add subsamples which are not local to each subsample
        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        aset['foo']._subsamples[50] = template
        aset[2]._subsamples[50] = template

        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)

            # test local only properties
            assert isinstance(sample.keys(local=True), Iterator)
            res = list(sample.keys(local=True))
            assert len(res) == len(subsample_data)
            for k in res:
                assert k in subsample_data
                assert k != 50

            # compare to local+remote properties
            assert isinstance(sample.keys(local=False), Iterator)
            res = list(sample.keys(local=False))
            assert len(res) == len(subsample_data) + 1
            assert 50 in res
            for k in res:
                assert k in list(subsample_data.keys()) + [50]

        del aset._samples['foo']._subsamples[50]
        del aset._samples[2]._subsamples[50]

    def test_get_sample_values_method(self, initialized_arrayset):
        from hangar.columns.layout_nested import FlatSubsampleReader
        from collections.abc import Iterator
        aset = initialized_arrayset

        assert isinstance(aset.values(), Iterator)
        res = list(aset.values())
        assert len(res) == 2
        for sample in res:
            assert sample.sample == 'foo' or 2
            assert isinstance(sample, FlatSubsampleReader)

    def test_get_sample_values_method_local_only(self, initialized_arrayset):
        from hangar.columns.layout_nested import FlatSubsampleReader
        from collections.abc import Iterator
        aset = initialized_arrayset
        # add subsamples which are not local to each subsample
        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        aset['foo']._subsamples[50] = template

        assert isinstance(aset.values(local=True), Iterator)
        res = list(aset.values(local=True))
        assert len(res) == 1
        sample = res[0]
        assert sample.sample == 2
        assert isinstance(sample, FlatSubsampleReader)

        del aset._samples['foo']._subsamples[50]

    def test_get_sample_subsample_values_method(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.values(), Iterator)
            res = list(sample.values())
            for v in res:
                assert any([np.allclose(v, arr) for arr in subsample_data.values()])

    def test_get_sample_subsample_values_method_local_only(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator
        aset = initialized_arrayset

        # add subsamples which are not local to each subsample
        # perform the mock
        from hangar.backends import backend_decoder
        from hangar.columns.common import open_file_handles
        template = backend_decoder(b'50:daeaaeeaebv')
        aset['foo']._subsamples[50] = template
        aset[2]._subsamples[50] = template
        mocked_fhand = open_file_handles(
            ['50'], path=initialized_arrayset._path, mode='a', schema=aset._schema)
        aset._be_fs['50'] = mocked_fhand['50']

        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)

            # test local only properties
            assert isinstance(sample.values(local=True), Iterator)
            res = list(sample.values(local=True))
            assert len(res) == len(subsample_data)
            for v in res:
                assert any([np.allclose(v, arr) for arr in subsample_data.values()])

            # test local+remote properties
            with pytest.raises(FileNotFoundError):
                list(sample.values(local=False))

        del aset._be_fs['50']
        del aset._samples['foo']._subsamples[50]
        del aset._samples[2]._subsamples[50]

    def test_get_sample_items_method(self, initialized_arrayset):
        from hangar.columns.layout_nested import FlatSubsampleReader
        from collections.abc import Iterator
        aset = initialized_arrayset

        assert isinstance(aset.items(), Iterator)
        res = list(aset.items())
        assert len(res) == 2
        for sample_name, sample in res:
            assert sample_name == 2 or 'foo'
            assert isinstance(sample, FlatSubsampleReader)
            assert sample_name == sample.sample

    def test_get_sample_items_method_local_only(self, initialized_arrayset):
        from hangar.columns.layout_nested import FlatSubsampleReader
        from collections.abc import Iterator
        aset = initialized_arrayset
        # add subsamples which are not local to each subsample
        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        aset['foo']._subsamples[50] = template

        assert isinstance(aset.items(local=True), Iterator)
        res = list(aset.items(local=True))
        assert len(res) == 1
        sample_name, sample = res[0]
        assert sample_name == 2
        assert isinstance(sample, FlatSubsampleReader)
        assert sample.sample == sample_name

        del aset._samples['foo']._subsamples[50]

    def test_get_sample_subsample_items_method(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.items(), Iterator)
            res = list(sample.items())
            for k, v in res:
                assert_equal(v, subsample_data[k])

    def test_get_sample_subsample_items_method_local_only(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator
        aset = initialized_arrayset

        # add subsamples which are not local to each subsample ato perform the mock
        from hangar.backends import backend_decoder
        from hangar.columns.common import open_file_handles
        template = backend_decoder(b'50:daeaaeeaebv')
        aset['foo']._subsamples[50] = template
        aset[2]._subsamples[50] = template
        mocked_fhand = open_file_handles(
            ['50'], path=initialized_arrayset._path, mode='a', schema=aset._schema)
        aset._be_fs['50'] = mocked_fhand['50']

        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)

            # test local only properties
            assert isinstance(sample.items(local=True), Iterator)
            res = list(sample.items(local=True))
            assert len(res) == len(subsample_data)
            for k, v in res:
                assert_equal(v, subsample_data[k])
                assert k != 50

            # test local+remote properties
            with pytest.raises(FileNotFoundError):
                list(sample.items(local=False))

        del aset._be_fs['50']
        del aset._samples['foo']._subsamples[50]
        del aset._samples[2]._subsamples[50]

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset3_backend", fixed_shape_backend_params)
    def test_arrayset_remote_references_property_with_none(
            self, aset1_backend, aset2_backend, aset3_backend, repo, randomsizedarray
    ):
        co = repo.checkout(write=True)
        aset1 = co.add_ndarray_column('aset1', prototype=randomsizedarray,
                                         backend=aset1_backend, contains_subsamples=True)
        aset2 = co.add_ndarray_column('aset2', shape=(2, 2), dtype=np.int,
                                         backend=aset2_backend, contains_subsamples=True)
        aset3 = co.add_ndarray_column('aset3', shape=(3, 4), dtype=np.float32,
                                         backend=aset3_backend, contains_subsamples=True)
        with aset1 as d1, aset2 as d2, aset3 as d3:
            d1[1] = {11: randomsizedarray}
            d2[1] = {21: np.ones((2, 2), dtype=np.int)}
            d3[1] = {31: np.ones((3, 4), dtype=np.float32)}

        assert co.columns.contains_remote_references == {'aset1': False, 'aset2': False, 'aset3': False}
        assert co.columns.remote_sample_keys == {'aset1': (), 'aset2': (), 'aset3': ()}
        co.close()

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset3_backend", fixed_shape_backend_params)
    def test_arrayset_remote_references_property_with_remotes(
            self, aset1_backend, aset2_backend, aset3_backend, repo, randomsizedarray
    ):
        co = repo.checkout(write=True)
        aset1 = co.add_ndarray_column('aset1', prototype=randomsizedarray,
                                         backend=aset1_backend, contains_subsamples=True)
        aset2 = co.add_ndarray_column('aset2', shape=(2, 2), dtype=np.int,
                                         backend=aset2_backend, contains_subsamples=True)
        aset3 = co.add_ndarray_column('aset3', shape=(3, 4), dtype=np.float32,
                                         backend=aset3_backend, contains_subsamples=True)
        with aset1 as d1, aset2 as d2, aset3 as d3:
            d1[1] = {11: randomsizedarray}
            d2[1] = {21: np.ones((2, 2), dtype=np.int)}
            d3[1] = {31: np.ones((3, 4), dtype=np.float32)}

        assert co.columns.contains_remote_references == {'aset1': False, 'aset2': False, 'aset3': False}
        assert co.columns.remote_sample_keys == {'aset1': (), 'aset2': (), 'aset3': ()}
        co.commit('hello')
        co.close()
        co = repo.checkout()
        # perform the mock
        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        co._columns._columns['aset1']._samples[1]._subsamples[12] = template
        co._columns._columns['aset2']._samples[1]._subsamples[22] = template

        assert co.columns.contains_remote_references == {'aset1': True, 'aset2': True, 'aset3': False}
        assert co.columns.remote_sample_keys == {'aset1': (1,), 'aset2': (1,), 'aset3': ()}
        co.close()


class TestWriteThenReadCheckout:

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_add_data_commit_checkout_read_only_contains_same(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('foo', shape=(5, 7), dtype=np.uint16,
                                        backend=backend, contains_subsamples=True)
        added = aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name, subsample_val in subsample_data.items():
                assert_equal(sample[subsample_name], subsample_val)
        co.commit('first')
        co.close()

        rco = repo.checkout()
        naset = rco.columns['foo']
        for sample_name, subsample_data in subsample_data_map.items():
            sample = naset.get(sample_name)
            for subsample_name, subsample_val in subsample_data.items():
                assert_equal(sample[subsample_name], subsample_val)
        rco.close()
