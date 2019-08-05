import pytest
import numpy as np


class TestCellstore(object):

    def test_invalid_dsetname(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        with pytest.raises(ValueError):
            co.cellstores.init_cellstore(name='invalid name', prototype=randomsizedarray)
        co.close()

    def test_read_only_mode(self, written_repo):
        import hangar
        co = written_repo.checkout()
        assert isinstance(co, hangar.checkout.ReaderCheckout)
        assert co.cellstores.init_cellstore is None
        assert co.cellstores.remove_dset is None
        assert len(co.cellstores['_dset']) == 0
        co.close()

    def test_get_cellstore(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)

        # getting the cellstore with `get`
        dsetOld = co.cellstores.get('_dset')
        dsetOldPath = dsetOld._path
        dsetOldDsetn = dsetOld._dsetn
        dsetOldDefaultSchemaHash = dsetOld._default_schema_hash

        dsetOld.add(array5by7, '1')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()

        # getting cellstore with dictionary like style method
        dsetNew = co.cellstores['_dset']

        assert np.allclose(dsetNew['1'], array5by7)
        assert dsetOldPath == dsetNew._path
        assert dsetOldDsetn == dsetNew._dsetn
        assert dsetOldDefaultSchemaHash == dsetNew._default_schema_hash
        co.close()

    @pytest.mark.parametrize("dset_backend", ['00', '10'])
    def test_remove_cellstore(self, dset_backend, written_repo):
        co = written_repo.checkout(write=True)
        co.cellstores.remove_dset('_dset')
        with pytest.raises(KeyError):
            co.cellstores.remove_dset('_dset')

        co.cellstores.init_cellstore(name='_dset', shape=(5, 7), dtype=np.float64, backend=dset_backend)
        assert len(co.cellstores) == 1
        co.cellstores.remove_dset('_dset')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        assert len(co.cellstores) == 0

        co.cellstores.init_cellstore(name='_dset', shape=(5, 7), dtype=np.float64, backend=dset_backend)
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout(write=True)
        assert len(co.cellstores) == 1
        del co.cellstores['_dset']
        assert len(co.cellstores) == 0
        co.commit('this is a commit message')
        co.close()

    @pytest.mark.parametrize("dset_backend", ['00', '10'])
    def test_init_again(self, dset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.cellstores.init_cellstore('dset', prototype=randomsizedarray, backend=dset_backend)
        with pytest.raises(LookupError):
            co.cellstores.init_cellstore('dset', prototype=randomsizedarray, backend=dset_backend)
        co.close()

    @pytest.mark.parametrize("dset_backend", ['00', '10'])
    def test_cellstore_with_more_dimension(self, dset_backend, repo):
        co = repo.checkout(write=True)
        shape = (0, 1, 2)
        with pytest.raises(ValueError):
            co.cellstores.init_cellstore('dset', shape=shape, dtype=np.int, backend=dset_backend)
        shape = [1] * 31
        dset = co.cellstores.init_cellstore('dset1', shape=shape, dtype=np.int, backend=dset_backend)
        assert len(dset._schema_max_shape) == 31
        shape = [1] * 32
        with pytest.raises(ValueError):
            # maximum tensor rank must be <= 31
            co.cellstores.init_cellstore('dset2', shape=shape, dtype=np.int, backend=dset_backend)
        co.close()

    @pytest.mark.parametrize("dset_backend", ['00', '10'])
    def test_cellstore_with_empty_dimension(self, dset_backend, repo):
        co = repo.checkout(write=True)
        arr = np.array(1, dtype=np.int64)
        dset = co.cellstores.init_cellstore('dset1', shape=(), dtype=np.int64, backend=dset_backend)
        dset['1'] = arr
        co.commit('this is a commit message')
        dset = co.cellstores.init_cellstore('dset2', prototype=arr)
        dset['1'] = arr
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        dset1 = co.cellstores['dset1']
        dset2 = co.cellstores['dset2']
        assert np.allclose(dset1['1'], arr)
        assert np.allclose(dset2['1'], arr)
        co.close()

    @pytest.mark.parametrize("dset_backend", ['00', '10'])
    def test_cellstore_with_int_specifier_as_dimension(self, dset_backend, repo):
        co = repo.checkout(write=True)
        arr = np.arange(10, dtype=np.int64)
        dset = co.cellstores.init_cellstore('dset1', shape=10, dtype=np.int64, backend=dset_backend)
        dset['1'] = arr
        co.commit('this is a commit message')
        arr2 = np.array(53, dtype=np.int64)
        dset = co.cellstores.init_cellstore('dset2', prototype=arr2)
        dset['1'] = arr2
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        dset1 = co.cellstores['dset1']
        dset2 = co.cellstores['dset2']
        assert np.allclose(dset1['1'], arr)
        assert np.allclose(dset2['1'], arr2)
        co.close()


class TestDataWithFixedSizedCellstore(object):

    @pytest.mark.parametrize("dset1_backend", ['00', '10'])
    @pytest.mark.parametrize("dset2_backend", ['00', '10'])
    @pytest.mark.parametrize("dset3_backend", ['00', '10'])
    def test_iterating_over(self, dset1_backend, dset2_backend, dset3_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        all_tensors = []
        dset1 = co.cellstores.init_cellstore('dset1', prototype=randomsizedarray, backend=dset1_backend)
        dset2 = co.cellstores.init_cellstore('dset2', shape=(2, 2), dtype=np.int, backend=dset2_backend)
        dset3 = co.cellstores.init_cellstore('dset3', shape=(3, 4), dtype=np.float32, backend=dset3_backend)

        with dset1 as d1, dset2 as d2, dset3 as d3:
            d1['1'] = randomsizedarray
            d1['2'] = np.zeros_like(randomsizedarray)
            d1['3'] = np.zeros_like(randomsizedarray) + 5

            d2['1'] = np.ones((2, 2), dtype=np.int)
            d2['2'] = np.ones((2, 2), dtype=np.int) * 5
            d2['3'] = np.zeros((2, 2), dtype=np.int)

            d3['1'] = np.ones((3, 4), dtype=np.float32)
            d3['2'] = np.ones((3, 4), dtype=np.float32) * 7
            d3['3'] = np.zeros((3, 4), dtype=np.float32)

        all_tensors.extend([dset1['1'], dset1['2'], dset1['3']])
        all_tensors.extend([dset2['1'], dset2['2'], dset2['3']])
        all_tensors.extend([dset3['1'], dset3['2'], dset3['3']])

        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        # iterating over .items()
        tensors_in_the_order = iter(all_tensors)
        for dname, dset in co.cellstores.items():
            assert dset._dsetn == dname
            for sname, sample in dset.items():
                assert np.allclose(sample, next(tensors_in_the_order))

        # iterating over .keys()
        tensors_in_the_order = iter(all_tensors)
        for dname in co.cellstores.keys():
            for sname in co.cellstores[dname].keys():
                assert np.allclose(co.cellstores[dname][sname], next(tensors_in_the_order))

        # iterating over .values()
        tensors_in_the_order = iter(all_tensors)
        for dset in co.cellstores.values():
            for sample in dset.values():
                assert np.allclose(sample, next(tensors_in_the_order))
        co.close()

    def test_get_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.cellstores['_dset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.cellstores['_dset']['1'], co.cellstores.get('_dset').get('1'), array5by7)
        co.close()

    def test_add_data_str_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dset = co.cellstores['_dset']
        with pytest.raises(KeyError):
            dset['somerandomkey']

        dset['1'] = array5by7
        dset.add(array5by7, '2')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.cellstores['_dset']['1'], array5by7)
        assert np.allclose(co.cellstores['_dset']['2'], array5by7)
        co.close()

    def test_add_data_int_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dset = co.cellstores['_dset']

        dset[1] = array5by7
        secondArray = array5by7 + 1
        dset.add(secondArray, 2)
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.cellstores['_dset'][1], array5by7)
        assert np.allclose(co.cellstores['_dset'][2], secondArray)
        co.close()

    def test_cannot_add_data_negative_int_key(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dset = co.cellstores['_dset']
        with pytest.raises(ValueError):
            dset[-1] = array5by7
        assert len(co.cellstores['_dset']) == 0
        co.close()

    def test_cannot_add_data_float_key(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dset = co.cellstores['_dset']
        with pytest.raises(ValueError):
            dset[2.1] = array5by7
        with pytest.raises(ValueError):
            dset[0.0] = array5by7
        assert len(co.cellstores['_dset']) == 0
        co.close()

    def test_add_data_mixed_int_str_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dset = co.cellstores['_dset']

        dset[1] = array5by7
        newFirstArray = array5by7 + 1
        dset['1'] = newFirstArray
        secondArray = array5by7 + 2
        dset.add(secondArray, 2)
        thirdArray = array5by7 + 3
        dset.add(thirdArray, '2')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.cellstores['_dset'][1], array5by7)
        assert np.allclose(co.cellstores['_dset']['1'], newFirstArray)
        assert np.allclose(co.cellstores['_dset'][2], secondArray)
        assert np.allclose(co.cellstores['_dset']['2'], thirdArray)
        co.close()

    def test_add_with_wrong_argument_order(self, w_checkout, array5by7):
        dset = w_checkout.cellstores['_dset']
        with pytest.raises(ValueError):
            dset.add('1', array5by7)

    def test_multiple_data_single_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.cellstores['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.cellstores['_dset']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        dset = co.cellstores['_dset']
        assert len(dset) == 2
        assert list(dset.keys()) == ['1', '2']
        assert np.allclose(dset['1'], array5by7)
        co.close()

    def test_multiple_data_multiple_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.cellstores['_dset'].add(array5by7, '1')
        co.commit('this is a commit message')
        new_array = np.zeros_like(array5by7)
        co.cellstores['_dset']['2'] = new_array
        co.close()

        new_new_array = new_array + 5
        co = written_repo.checkout(write=True)
        co.cellstores['_dset']['3'] = new_new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        dset = co.cellstores['_dset']
        assert np.allclose(dset['1'], array5by7)
        assert np.allclose(dset['2'], new_array)
        assert np.allclose(dset['3'], new_new_array)
        co.close()

    def test_added_but_not_commited(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.cellstores['_dset'].add(array5by7, '1')
        co.close()

        with pytest.raises(PermissionError):
            co.commit('this is a commit message')

        co = written_repo.checkout()
        dset = co.cellstores['_dset']
        with pytest.raises(KeyError):
            dset['1']
        co.close()

    def test_remove_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.cellstores['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.cellstores['_dset']['2'] = new_array
        co.cellstores['_dset']['3'] = new_array + 5
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        co.cellstores['_dset'].remove('1')
        del co.cellstores['_dset']['3']
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        with pytest.raises(KeyError):
            co.cellstores['_dset']['1']
        with pytest.raises(KeyError):
            co.cellstores['_dset']['3']
        assert len(co.cellstores['_dset']) == 1
        assert np.allclose(co.cellstores['_dset']['2'], new_array)
        co.close()

    def test_remove_all_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.cellstores['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.cellstores['_dset']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        co.cellstores['_dset'].remove('1')
        co.cellstores['_dset'].remove('2')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        with pytest.raises(KeyError):
            # removal of all data removes the cellstore
            co.cellstores['_dset']
        co.close()

        # recreating same and verifying
        co = written_repo.checkout(write=True)
        co.cellstores.init_cellstore('_dset', prototype=array5by7)
        co.cellstores['_dset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.cellstores['_dset']['1'], array5by7)
        co.close()

    @pytest.mark.parametrize("dset1_backend", ['00', '10'])
    @pytest.mark.parametrize("dset2_backend", ['00', '10'])
    def test_multiple_cellstores_single_commit(self, dset1_backend, dset2_backend, written_repo, randomsizedarray):
        co = written_repo.checkout(write=True)
        dset1 = co.cellstores.init_cellstore('dset1', prototype=randomsizedarray, backend=dset1_backend)
        dset2 = co.cellstores.init_cellstore('dset2', prototype=randomsizedarray, backend=dset2_backend)
        dset1['arr'] = randomsizedarray
        dset2['arr'] = randomsizedarray
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.cellstores['dset1']['arr'], randomsizedarray)
        assert np.allclose(co.cellstores['dset2']['arr'], randomsizedarray)
        co.close()

    @pytest.mark.parametrize("dset1_backend", ['00', '10'])
    @pytest.mark.parametrize("dset2_backend", ['00', '10'])
    def test_prototype_and_shape(self, dset1_backend, dset2_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset1 = co.cellstores.init_cellstore(
            'dset1', prototype=randomsizedarray, backend=dset1_backend)
        dset2 = co.cellstores.init_cellstore(
            'dset2', shape=randomsizedarray.shape, dtype=randomsizedarray.dtype, backend=dset2_backend)

        newarray = np.random.random(randomsizedarray.shape).astype(randomsizedarray.dtype)
        dset1['arr1'] = newarray
        dset2['arr'] = newarray
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert np.allclose(co.cellstores['dset1']['arr1'], newarray)
        assert np.allclose(co.cellstores['dset2']['arr'], newarray)
        co.close()

    def test_samples_without_name(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset = co.cellstores.init_cellstore('dset', prototype=randomsizedarray)
        with pytest.raises(ValueError):
            dset.add(randomsizedarray)

        dset_no_name = co.cellstores.init_cellstore('dset_no_name',
                                                prototype=randomsizedarray,
                                                named_samples=False)
        dset_no_name.add(randomsizedarray)
        assert np.allclose(next(dset_no_name.values()), randomsizedarray)
        co.close()

    def test_different_data_types_and_shapes(self, repo):
        co = repo.checkout(write=True)
        shape = (2, 3)
        dtype = np.int
        another_dtype = np.float64
        another_shape = (3, 4)
        arr = np.random.random(shape).astype(dtype)
        dset = co.cellstores.init_cellstore('dset', shape=shape, dtype=dtype)
        dset['1'] = arr

        newarr = np.random.random(shape).astype(another_dtype)
        with pytest.raises(ValueError):
            dset['2'] = newarr

        newarr = np.random.random(another_shape).astype(dtype)
        with pytest.raises(ValueError):
            dset['3'] = newarr
        co.close()

    @pytest.mark.parametrize("dset_backend", ['00', '10'])
    def test_adding_same_data_again_with_same_name(self, dset_backend, repo, array5by7):
        co = repo.checkout(write=True)
        dset = co.cellstores.init_cellstore('dset', prototype=array5by7, backend=dset_backend)
        dset['1'] = array5by7
        with pytest.raises(LookupError):
            # raises before commit
            dset['1'] = array5by7
        co.commit('this is a commit message')
        with pytest.raises(LookupError):
            # raises after commit
            dset['1'] = array5by7
        co.close()
        co = repo.checkout(write=True)
        dset = co.cellstores['dset']
        with pytest.raises(LookupError):
            # raises in another checkout
            dset['1'] = array5by7
        co.close()

    @pytest.mark.parametrize("dset_backend", ['00', '10'])
    def test_writer_context_manager_cellstore_add_sample(self, dset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset = co.cellstores.init_cellstore('dset', prototype=randomsizedarray, backend=dset_backend)
        with co.cellstores['dset'] as dset:
            dset.add(randomsizedarray, '1')
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        assert np.allclose(co.cellstores['dset']['1'], randomsizedarray)
        co.close()

    def test_writer_context_manager_metadata_add(self, repo):
        co = repo.checkout(write=True)
        with co.metadata as metadata:
            metadata.add('key', 'val')
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        assert co.metadata['key'] == 'val'
        co.close()

    @pytest.mark.parametrize("dset_backend", ['00', '10'])
    def test_cellstore_context_manager_dset_sample_and_metadata_add(self, dset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset = co.cellstores.init_cellstore('dset', prototype=randomsizedarray, backend=dset_backend)
        with co.cellstores['dset'] as dset:
            dset.add(randomsizedarray, '1')
            co.metadata['hello'] = 'world'
        with co.metadata as metadata:
            newarr = randomsizedarray + 1
            dset['2'] = newarr
            metadata.add('key', 'val')
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert np.allclose(co.cellstores['dset']['1'], randomsizedarray)
        assert np.allclose(co.cellstores['dset'].get('2'), newarr)
        assert co.metadata['key'] == 'val'
        assert co.metadata.get('hello') == 'world'
        co.close()

    @pytest.mark.parametrize("dset1_backend", ['00', '10'])
    @pytest.mark.parametrize("dset2_backend", ['00', '10'])
    def test_bulk_add(self, dset1_backend, dset2_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.cellstores.init_cellstore(
            'dset_no_name1',
            prototype=randomsizedarray,
            named_samples=False,
            backend=dset1_backend)
        co.cellstores.init_cellstore(
            'dset_no_name2',
            prototype=randomsizedarray,
            named_samples=False,
            backend=dset2_backend)
        co.commit('this is a commit message')

        # dummy additino with wrong key
        with pytest.raises(KeyError):
            co.cellstores.multi_add(
                {
                    'dset_no_name2': randomsizedarray / 255,
                    'dummykey': randomsizedarray
                })
        # making sure above addition did not add partial data
        with pytest.raises(RuntimeError):
            co.commit('this is a commit message')

        # proper addition and verification
        co.cellstores.multi_add(
            {
                'dset_no_name1': randomsizedarray,
                'dset_no_name2': randomsizedarray / 255
            })
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        data1 = next(co.cellstores['dset_no_name1'].values())
        data2 = next(co.cellstores['dset_no_name2'].values())
        assert np.allclose(data1, randomsizedarray)
        assert np.allclose(data2, randomsizedarray / 255)
        co.close()

    def test_writer_cellstore_properties_are_correct(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        d = co.cellstores['_dset']
        assert d.name == '_dset'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.variable_shape is False
        assert d.named_samples is True
        assert d.iswriteable is True
        co.close()

    def test_reader_cellstore_properties_are_correct(self, written_repo, array5by7):
        co = written_repo.checkout(write=False)
        d = co.cellstores['_dset']
        assert d.name == '_dset'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.variable_shape is False
        assert d.named_samples is True
        assert d.iswriteable is False


class TestVariableSizedCellstore(object):

    @pytest.mark.parametrize(
        'test_shapes,shape',
        [[[(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10)],
         [[(10,), (1,), (5,)], (10,)],
         [[(100, 100, 100), (100, 100, 1), (100, 1, 100), (1, 100, 100), (1, 1, 1), (34, 6, 3)], (100, 100, 100)]])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_writer_can_create_variable_size_cellstore(self, written_repo, dtype, test_shapes, shape, backend):
        repo = written_repo
        wco = repo.checkout(write=True)
        wco.cellstores.init_cellstore('vardset', shape=shape, dtype=dtype, variable_shape=True, backend=backend)
        d = wco.cellstores['vardset']

        arrdict = {}
        for idx, shape in enumerate(test_shapes):
            arr = (np.random.random_sample(shape) * 10).astype(dtype)
            arrdict[str(idx)] = arr
            d[str(idx)] = arr

        for k, v in arrdict.items():
            # make sure they are good before committed
            assert np.allclose(d[k], v)

        wco.commit('first')

        for k, v in arrdict.items():
            # make sure they can work after commit
            assert np.allclose(d[k], v)
        wco.close()

    @pytest.mark.parametrize('test_shapes,shape', [
        [[(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10)],
        [[(10,), (1,), (5,)], (10,)],
        [[(100, 100, 100), (100, 100, 1), (100, 1, 100), (1, 100, 100), (1, 1, 1), (34, 6, 3)], (100, 100, 100)]
    ])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_reader_recieves_expected_values_for_variable_size_cellstore(self, written_repo, dtype, test_shapes, shape, backend):
        repo = written_repo
        wco = repo.checkout(write=True)
        wco.cellstores.init_cellstore('vardset', shape=shape, dtype=dtype, variable_shape=True, backend=backend)
        wd = wco.cellstores['vardset']

        arrdict = {}
        for idx, shape in enumerate(test_shapes):
            arr = (np.random.random_sample(shape) * 10).astype(dtype)
            arrdict[str(idx)] = arr
            wd[str(idx)] = arr

        for k, v in arrdict.items():
            # make sure they are good before committed
            assert np.allclose(wd[k], v)

        wco.commit('first')
        rco = repo.checkout()
        rd = rco.cellstores['vardset']

        for k, v in arrdict.items():
            # make sure they can work after commit
            assert np.allclose(wd[k], v)
            assert np.allclose(rd[k], v)
        wco.close()
        rco.close()


    @pytest.mark.parametrize('dset_specs', [
        [['dset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.uint8],
         ['dset2', [(10,), (1,), (5,)], (10,), '00', np.uint8]
         ],
        [['dset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.uint8],
         ['dset2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['dset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.uint8],
         ['dset2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['dset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.uint8],
         ['dset2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['dset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.float32],
         ['dset2', [(10,), (1,), (5,)], (10,), '00', np.float32]
         ],
        [['dset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.float32],
         ['dset2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ],
        [['dset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.float32],
         ['dset2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ],
        [['dset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.float32],
         ['dset2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ]])
    def test_writer_reader_can_create_read_multiple_variable_size_cellstore(self, written_repo, dset_specs):
        repo = written_repo
        wco = repo.checkout(write=True)
        arrdict = {}
        for dset_spec in dset_specs:
            dset_name, test_shapes, max_shape, backend, dtype = dset_spec
            wco.cellstores.init_cellstore(dset_name, shape=max_shape, dtype=dtype, variable_shape=True, backend=backend)

            arrdict[dset_name] = {}
            for idx, shape in enumerate(test_shapes):
                arr = (np.random.random_sample(shape) * 10).astype(dtype)
                arrdict[dset_name][str(idx)] = arr
                wco.cellstores[dset_name][str(idx)] = arr

        for dset_k in arrdict.keys():
            for samp_k, v in arrdict[dset_k].items():
                # make sure they are good before committed
                assert np.allclose(wco.cellstores[dset_k][samp_k], v)

        wco.commit('first')
        rco = repo.checkout()

        for dset_k in arrdict.keys():
            for samp_k, v in arrdict[dset_k].items():
                # make sure they are good before committed
                assert np.allclose(wco.cellstores[dset_k][samp_k], v)
                assert np.allclose(rco.cellstores[dset_k][samp_k], v)
        wco.close()
        rco.close()

    def test_writer_cellstore_properties_are_correct(self, variable_shape_written_repo):
        co = variable_shape_written_repo.checkout(write=True)
        d = co.cellstores['_dset']
        assert d.name == '_dset'
        assert d.dtype == np.float64
        assert np.allclose(d.shape, (10, 10))
        assert d.variable_shape is True
        assert d.named_samples is True
        assert d.iswriteable is True
        co.close()

    def test_reader_cellstore_properties_are_correct(self, variable_shape_written_repo):
        co = variable_shape_written_repo.checkout(write=False)
        d = co.cellstores['_dset']
        assert d.name == '_dset'
        assert d.dtype == np.float64
        assert np.allclose(d.shape, (10, 10))
        assert d.variable_shape is True
        assert d.named_samples is True
        assert d.iswriteable is False
        co.close()


class TestMultiprocessCellstoreReads(object):

    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_external_multi_process_pool(self, repo, backend):
        from multiprocessing import get_context

        masterCmtList = []
        co = repo.checkout(write=True)
        co.cellstores.init_cellstore(name='_dset', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        for cIdx in range(2):
            if cIdx != 0:
                co = repo.checkout(write=True)
            with co.cellstores['_dset'] as d:
                kstart = 20 * cIdx
                for sIdx in range(20):
                    arr = np.random.randn(20, 20).astype(np.float32) * 100
                    sName = str(sIdx + kstart)
                    d[sName] = arr
                    masterSampList.append(arr)
            assert d._backend == backend
            cmt = co.commit(f'master commit number: {cIdx}')
            masterCmtList.append((cmt, list(masterSampList)))
            co.close()

        cmtIdx = 0
        for cmt, sampList in masterCmtList:
            nco = repo.checkout(write=False, commit=cmt)
            ds = nco.cellstores['_dset']
            keys = [str(i) for i in range(20 + (20*cmtIdx))]
            with get_context('spawn').Pool(2) as P:
                cmtData = P.map(ds.get, keys)
            for data, sampData in zip(cmtData, sampList):
                assert np.allclose(data, sampData) is True
            cmtIdx += 1
            nco.close()

    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_batch_get_multi_process_pool(self, repo, backend):
        masterCmtList = []
        co = repo.checkout(write=True)
        co.cellstores.init_cellstore(name='_dset', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        for cIdx in range(2):
            if cIdx != 0:
                co = repo.checkout(write=True)
            with co.cellstores['_dset'] as d:
                kstart = 20 * cIdx
                for sIdx in range(20):
                    arr = np.random.randn(20, 20).astype(np.float32) * 100
                    sName = str(sIdx + kstart)
                    d[sName] = arr
                    masterSampList.append(arr)
                assert d._backend == backend
            cmt = co.commit(f'master commit number: {cIdx}')
            masterCmtList.append((cmt, list(masterSampList)))
            co.close()

        cmtIdx = 0
        for cmt, sampList in masterCmtList:
            nco = repo.checkout(write=False, commit=cmt)
            ds = nco.cellstores['_dset']
            keys = [str(i) for i in range(20 + (20*cmtIdx))]
            cmtData = ds.get_batch(keys, n_cpus=2)
            for data, sampData in zip(cmtData, sampList):
                assert np.allclose(data, sampData) is True
            cmtIdx += 1
            nco.close()

    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_batch_get_fails_on_superset_of_keys_and_succeeds_on_subset(self, repo, backend):
        co = repo.checkout(write=True)
        co.cellstores.init_cellstore(name='_dset', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        with co.cellstores['_dset'] as d:
            for sIdx in range(20):
                arr = np.random.randn(20, 20).astype(np.float32) * 100
                sName = str(sIdx)
                d[sName] = arr
                masterSampList.append(arr)
            assert d._backend == backend
        cmt = co.commit(f'master commit number one')
        co.close()

        nco = repo.checkout(write=False, commit=cmt)
        ds = nco.cellstores['_dset']

        # superset of keys fails
        with pytest.raises(KeyError):
            keys = [str(i) for i in range(24)]
            ds.get_batch(keys, n_cpus=2)

        # subset of keys works
        keys = [str(i) for i in range(10, 20)]
        cmtData = ds.get_batch(keys, n_cpus=2)
        for idx, data in enumerate(cmtData):
            assert np.allclose(data, masterSampList[10+idx]) is True
        nco.close()

    def test_writer_iterating_over_keys_can_have_additions_made_no_error(self, written_two_cmt_repo):
        # do not want ``RuntimeError dictionary changed size during iteration``

        repo = written_two_cmt_repo
        co = repo.checkout(write=True)
        dset = co.cellstores['_dset']
        klist = []
        with dset as ds:
            for idx, k in enumerate(ds.keys()):
                klist.append(k)
                if idx == 0:
                    ds['1232'] = np.random.randn(5, 7).astype(np.float32)
        assert '1232' not in klist

        klist = []
        for k in ds.keys():
            klist.append(k)
        assert '1232' in klist
        co.close()

    def test_writer_iterating_over_values_can_have_additions_made_no_error(self, written_two_cmt_repo):
        # do not want ``RuntimeError dictionary changed size during iteration``

        repo = written_two_cmt_repo
        co = repo.checkout(write=True)
        dset = co.cellstores['_dset']
        vlist = []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with dset as ds:
            for idx, v in enumerate(ds.values()):
                assert not np.allclose(v, mysample)
                vlist.append(v)
                if idx == 0:
                    ds['1232'] = mysample

        has_been_seen = []
        for v in ds.values():
            has_been_seen.append(np.allclose(v, mysample))

        assert any(has_been_seen) is True
        assert has_been_seen.count(True) == 1
        co.close()

    def test_writer_iterating_over_items_can_have_additions_made_no_error(self, written_two_cmt_repo):
        # do not want ``RuntimeError dictionary changed size during iteration``

        repo = written_two_cmt_repo
        co = repo.checkout(write=True)
        dset = co.cellstores['_dset']
        vlist, klist = [], []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with dset as ds:
            for idx, kv in enumerate(ds.items()):
                k, v = kv
                assert not np.allclose(v, mysample)
                vlist.append(v)
                klist.append(k)
                if idx == 0:
                    ds['1232'] = mysample

        assert '1232' not in klist
        khas_been_seen = []
        vhas_been_seen = []
        for k, v in ds.items():
            khas_been_seen.append(bool(k == '1232'))
            vhas_been_seen.append(np.allclose(v, mysample))

        assert any(khas_been_seen) is True
        assert khas_been_seen.count(True) == 1
        assert any(vhas_been_seen) is True
        assert vhas_been_seen.count(True) == 1
        co.close()

    def test_reader_iterating_over_items_can_not_make_additions(self, written_two_cmt_repo):
        # do not want ``RuntimeError dictionary changed size during iteration``

        repo = written_two_cmt_repo
        co = repo.checkout(write=False)
        dset = co.cellstores['_dset']
        vlist, klist = [], []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with dset as ds:
            for idx, kv in enumerate(ds.items()):
                k, v = kv
                assert not np.allclose(v, mysample)
                vlist.append(v)
                klist.append(k)
                if idx == 0:
                    with pytest.raises(TypeError):
                        ds['1232'] = mysample

        assert '1232' not in klist
        co.close()
