import pytest
import numpy as np


class TestArrayset(object):

    def test_invalid_asetname(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        with pytest.raises(ValueError):
            co.arraysets.init_arrayset(name='invalid name', prototype=randomsizedarray)
        co.close()

    def test_read_only_mode(self, written_repo):
        import hangar
        co = written_repo.checkout()
        assert isinstance(co, hangar.checkout.ReaderCheckout)
        assert co.arraysets.init_arrayset is None
        assert co.arraysets.remove_aset is None
        assert len(co.arraysets['_aset']) == 0
        co.close()

    def test_get_arrayset(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)

        # getting the arrayset with `get`
        asetOld = co.arraysets.get('_aset')
        asetOldPath = asetOld._path
        asetOldAsetn = asetOld._asetn
        asetOldDefaultSchemaHash = asetOld._default_schema_hash

        asetOld.add(array5by7, '1')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()

        # getting arrayset with dictionary like style method
        asetNew = co.arraysets['_aset']

        assert np.allclose(asetNew['1'], array5by7)
        assert asetOldPath == asetNew._path
        assert asetOldAsetn == asetNew._asetn
        assert asetOldDefaultSchemaHash == asetNew._default_schema_hash
        co.close()

    @pytest.mark.parametrize("aset_backend", ['00', '10'])
    def test_remove_arrayset(self, aset_backend, written_repo):
        co = written_repo.checkout(write=True)
        co.arraysets.remove_aset('_aset')
        with pytest.raises(KeyError):
            co.arraysets.remove_aset('_aset')

        co.arraysets.init_arrayset(name='_aset', shape=(5, 7), dtype=np.float64, backend=aset_backend)
        assert len(co.arraysets) == 1
        co.arraysets.remove_aset('_aset')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        assert len(co.arraysets) == 0

        co.arraysets.init_arrayset(name='_aset', shape=(5, 7), dtype=np.float64, backend=aset_backend)
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout(write=True)
        assert len(co.arraysets) == 1
        del co.arraysets['_aset']
        assert len(co.arraysets) == 0
        co.commit('this is a commit message')
        co.close()

    @pytest.mark.parametrize("aset_backend", ['00', '10'])
    def test_init_again(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.arraysets.init_arrayset('aset', prototype=randomsizedarray, backend=aset_backend)
        with pytest.raises(LookupError):
            co.arraysets.init_arrayset('aset', prototype=randomsizedarray, backend=aset_backend)
        co.close()

    @pytest.mark.parametrize("aset_backend", ['00', '10'])
    def test_arrayset_with_more_dimension(self, aset_backend, repo):
        co = repo.checkout(write=True)
        shape = (0, 1, 2)
        with pytest.raises(ValueError):
            co.arraysets.init_arrayset('aset', shape=shape, dtype=np.int, backend=aset_backend)
        shape = [1] * 31
        aset = co.arraysets.init_arrayset('aset1', shape=shape, dtype=np.int, backend=aset_backend)
        assert len(aset._schema_max_shape) == 31
        shape = [1] * 32
        with pytest.raises(ValueError):
            # maximum tensor rank must be <= 31
            co.arraysets.init_arrayset('aset2', shape=shape, dtype=np.int, backend=aset_backend)
        co.close()

    @pytest.mark.parametrize("aset_backend", ['00', '10'])
    def test_arrayset_with_empty_dimension(self, aset_backend, repo):
        co = repo.checkout(write=True)
        arr = np.array(1, dtype=np.int64)
        aset = co.arraysets.init_arrayset('aset1', shape=(), dtype=np.int64, backend=aset_backend)
        aset['1'] = arr
        co.commit('this is a commit message')
        aset = co.arraysets.init_arrayset('aset2', prototype=arr)
        aset['1'] = arr
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        aset1 = co.arraysets['aset1']
        aset2 = co.arraysets['aset2']
        assert np.allclose(aset1['1'], arr)
        assert np.allclose(aset2['1'], arr)
        co.close()

    @pytest.mark.parametrize("aset_backend", ['00', '10'])
    def test_arrayset_with_int_specifier_as_dimension(self, aset_backend, repo):
        co = repo.checkout(write=True)
        arr = np.arange(10, dtype=np.int64)
        aset = co.arraysets.init_arrayset('aset1', shape=10, dtype=np.int64, backend=aset_backend)
        aset['1'] = arr
        co.commit('this is a commit message')
        arr2 = np.array(53, dtype=np.int64)
        aset = co.arraysets.init_arrayset('aset2', prototype=arr2)
        aset['1'] = arr2
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        aset1 = co.arraysets['aset1']
        aset2 = co.arraysets['aset2']
        assert np.allclose(aset1['1'], arr)
        assert np.allclose(aset2['1'], arr2)
        co.close()


class TestDataWithFixedSizedArrayset(object):

    @pytest.mark.parametrize("aset1_backend", ['00', '10'])
    @pytest.mark.parametrize("aset2_backend", ['00', '10'])
    @pytest.mark.parametrize("aset3_backend", ['00', '10'])
    def test_iterating_over(self, aset1_backend, aset2_backend, aset3_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        all_tensors = []
        aset1 = co.arraysets.init_arrayset('aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.arraysets.init_arrayset('aset2', shape=(2, 2), dtype=np.int, backend=aset2_backend)
        aset3 = co.arraysets.init_arrayset('aset3', shape=(3, 4), dtype=np.float32, backend=aset3_backend)

        with aset1 as d1, aset2 as d2, aset3 as d3:
            d1['1'] = randomsizedarray
            d1['2'] = np.zeros_like(randomsizedarray)
            d1['3'] = np.zeros_like(randomsizedarray) + 5

            d2['1'] = np.ones((2, 2), dtype=np.int)
            d2['2'] = np.ones((2, 2), dtype=np.int) * 5
            d2['3'] = np.zeros((2, 2), dtype=np.int)

            d3['1'] = np.ones((3, 4), dtype=np.float32)
            d3['2'] = np.ones((3, 4), dtype=np.float32) * 7
            d3['3'] = np.zeros((3, 4), dtype=np.float32)

        all_tensors.extend([aset1['1'], aset1['2'], aset1['3']])
        all_tensors.extend([aset2['1'], aset2['2'], aset2['3']])
        all_tensors.extend([aset3['1'], aset3['2'], aset3['3']])

        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        # iterating over .items()
        tensors_in_the_order = iter(all_tensors)
        for dname, aset in co.arraysets.items():
            assert aset._asetn == dname
            for sname, sample in aset.items():
                assert np.allclose(sample, next(tensors_in_the_order))

        # iterating over .keys()
        tensors_in_the_order = iter(all_tensors)
        for dname in co.arraysets.keys():
            for sname in co.arraysets[dname].keys():
                assert np.allclose(co.arraysets[dname][sname], next(tensors_in_the_order))

        # iterating over .values()
        tensors_in_the_order = iter(all_tensors)
        for aset in co.arraysets.values():
            for sample in aset.values():
                assert np.allclose(sample, next(tensors_in_the_order))
        co.close()

    def test_get_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.arraysets['_aset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.arraysets['_aset']['1'], co.arraysets.get('_aset').get('1'), array5by7)
        co.close()

    def test_add_data_str_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        aset = co.arraysets['_aset']
        with pytest.raises(KeyError):
            aset['somerandomkey']

        aset['1'] = array5by7
        aset.add(array5by7, '2')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.arraysets['_aset']['1'], array5by7)
        assert np.allclose(co.arraysets['_aset']['2'], array5by7)
        co.close()

    def test_add_data_int_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        aset = co.arraysets['_aset']

        aset[1] = array5by7
        secondArray = array5by7 + 1
        aset.add(secondArray, 2)
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.arraysets['_aset'][1], array5by7)
        assert np.allclose(co.arraysets['_aset'][2], secondArray)
        co.close()

    def test_cannot_add_data_negative_int_key(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        aset = co.arraysets['_aset']
        with pytest.raises(ValueError):
            aset[-1] = array5by7
        assert len(co.arraysets['_aset']) == 0
        co.close()

    def test_cannot_add_data_float_key(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        aset = co.arraysets['_aset']
        with pytest.raises(ValueError):
            aset[2.1] = array5by7
        with pytest.raises(ValueError):
            aset[0.0] = array5by7
        assert len(co.arraysets['_aset']) == 0
        co.close()

    def test_add_data_mixed_int_str_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        aset = co.arraysets['_aset']

        aset[1] = array5by7
        newFirstArray = array5by7 + 1
        aset['1'] = newFirstArray
        secondArray = array5by7 + 2
        aset.add(secondArray, 2)
        thirdArray = array5by7 + 3
        aset.add(thirdArray, '2')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.arraysets['_aset'][1], array5by7)
        assert np.allclose(co.arraysets['_aset']['1'], newFirstArray)
        assert np.allclose(co.arraysets['_aset'][2], secondArray)
        assert np.allclose(co.arraysets['_aset']['2'], thirdArray)
        co.close()

    def test_add_with_wrong_argument_order(self, w_checkout, array5by7):
        aset = w_checkout.arraysets['_aset']
        with pytest.raises(ValueError):
            aset.add('1', array5by7)

    def test_multiple_data_single_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.arraysets['_aset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.arraysets['_aset']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        aset = co.arraysets['_aset']
        assert len(aset) == 2
        assert list(aset.keys()) == ['1', '2']
        assert np.allclose(aset['1'], array5by7)
        co.close()

    def test_multiple_data_multiple_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.arraysets['_aset'].add(array5by7, '1')
        co.commit('this is a commit message')
        new_array = np.zeros_like(array5by7)
        co.arraysets['_aset']['2'] = new_array
        co.close()

        new_new_array = new_array + 5
        co = written_repo.checkout(write=True)
        co.arraysets['_aset']['3'] = new_new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        aset = co.arraysets['_aset']
        assert np.allclose(aset['1'], array5by7)
        assert np.allclose(aset['2'], new_array)
        assert np.allclose(aset['3'], new_new_array)
        co.close()

    def test_added_but_not_commited(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.arraysets['_aset'].add(array5by7, '1')
        co.close()

        with pytest.raises(PermissionError):
            co.commit('this is a commit message')

        co = written_repo.checkout()
        aset = co.arraysets['_aset']
        with pytest.raises(KeyError):
            aset['1']
        co.close()

    def test_remove_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.arraysets['_aset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.arraysets['_aset']['2'] = new_array
        co.arraysets['_aset']['3'] = new_array + 5
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        co.arraysets['_aset'].remove('1')
        del co.arraysets['_aset']['3']
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        with pytest.raises(KeyError):
            co.arraysets['_aset']['1']
        with pytest.raises(KeyError):
            co.arraysets['_aset']['3']
        assert len(co.arraysets['_aset']) == 1
        assert np.allclose(co.arraysets['_aset']['2'], new_array)
        co.close()

    def test_remove_all_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.arraysets['_aset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.arraysets['_aset']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        co.arraysets['_aset'].remove('1')
        co.arraysets['_aset'].remove('2')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        with pytest.raises(KeyError):
            # removal of all data removes the arrayset
            co.arraysets['_aset']
        co.close()

        # recreating same and verifying
        co = written_repo.checkout(write=True)
        co.arraysets.init_arrayset('_aset', prototype=array5by7)
        co.arraysets['_aset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.arraysets['_aset']['1'], array5by7)
        co.close()

    @pytest.mark.parametrize("aset1_backend", ['00', '10'])
    @pytest.mark.parametrize("aset2_backend", ['00', '10'])
    def test_multiple_arraysets_single_commit(self, aset1_backend, aset2_backend, written_repo, randomsizedarray):
        co = written_repo.checkout(write=True)
        aset1 = co.arraysets.init_arrayset('aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.arraysets.init_arrayset('aset2', prototype=randomsizedarray, backend=aset2_backend)
        aset1['arr'] = randomsizedarray
        aset2['arr'] = randomsizedarray
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.arraysets['aset1']['arr'], randomsizedarray)
        assert np.allclose(co.arraysets['aset2']['arr'], randomsizedarray)
        co.close()

    @pytest.mark.parametrize("aset1_backend", ['00', '10'])
    @pytest.mark.parametrize("aset2_backend", ['00', '10'])
    def test_prototype_and_shape(self, aset1_backend, aset2_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset1 = co.arraysets.init_arrayset(
            'aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.arraysets.init_arrayset(
            'aset2', shape=randomsizedarray.shape, dtype=randomsizedarray.dtype, backend=aset2_backend)

        newarray = np.random.random(randomsizedarray.shape).astype(randomsizedarray.dtype)
        aset1['arr1'] = newarray
        aset2['arr'] = newarray
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert np.allclose(co.arraysets['aset1']['arr1'], newarray)
        assert np.allclose(co.arraysets['aset2']['arr'], newarray)
        co.close()

    def test_samples_without_name(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('aset', prototype=randomsizedarray)
        with pytest.raises(ValueError):
            aset.add(randomsizedarray)

        aset_no_name = co.arraysets.init_arrayset('aset_no_name',
                                                prototype=randomsizedarray,
                                                named_samples=False)
        aset_no_name.add(randomsizedarray)
        assert np.allclose(next(aset_no_name.values()), randomsizedarray)
        co.close()

    def test_different_data_types_and_shapes(self, repo):
        co = repo.checkout(write=True)
        shape = (2, 3)
        dtype = np.int
        another_dtype = np.float64
        another_shape = (3, 4)
        arr = np.random.random(shape).astype(dtype)
        aset = co.arraysets.init_arrayset('aset', shape=shape, dtype=dtype)
        aset['1'] = arr

        newarr = np.random.random(shape).astype(another_dtype)
        with pytest.raises(ValueError):
            aset['2'] = newarr

        newarr = np.random.random(another_shape).astype(dtype)
        with pytest.raises(ValueError):
            aset['3'] = newarr
        co.close()

    @pytest.mark.parametrize("aset_backend", ['00', '10'])
    def test_writer_context_manager_arrayset_add_sample(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('aset', prototype=randomsizedarray, backend=aset_backend)
        with co.arraysets['aset'] as aset:
            aset.add(randomsizedarray, '1')
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        assert np.allclose(co.arraysets['aset']['1'], randomsizedarray)
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

    @pytest.mark.parametrize("aset_backend", ['00', '10'])
    def test_arrayset_context_manager_aset_sample_and_metadata_add(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('aset', prototype=randomsizedarray, backend=aset_backend)
        with co.arraysets['aset'] as aset:
            aset.add(randomsizedarray, '1')
            co.metadata['hello'] = 'world'
        with co.metadata as metadata:
            newarr = randomsizedarray + 1
            aset['2'] = newarr
            metadata.add('key', 'val')
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert np.allclose(co.arraysets['aset']['1'], randomsizedarray)
        assert np.allclose(co.arraysets['aset'].get('2'), newarr)
        assert co.metadata['key'] == 'val'
        assert co.metadata.get('hello') == 'world'
        co.close()

    @pytest.mark.parametrize("aset1_backend", ['00', '10'])
    @pytest.mark.parametrize("aset2_backend", ['00', '10'])
    def test_bulk_add(self, aset1_backend, aset2_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.arraysets.init_arrayset(
            'aset_no_name1',
            prototype=randomsizedarray,
            named_samples=False,
            backend=aset1_backend)
        co.arraysets.init_arrayset(
            'aset_no_name2',
            prototype=randomsizedarray,
            named_samples=False,
            backend=aset2_backend)
        co.commit('this is a commit message')

        # dummy additino with wrong key
        with pytest.raises(KeyError):
            co.arraysets.multi_add(
                {
                    'aset_no_name2': randomsizedarray / 255,
                    'dummykey': randomsizedarray
                })
        # making sure above addition did not add partial data
        with pytest.raises(RuntimeError):
            co.commit('this is a commit message')

        # proper addition and verification
        co.arraysets.multi_add(
            {
                'aset_no_name1': randomsizedarray,
                'aset_no_name2': randomsizedarray / 255
            })
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        data1 = next(co.arraysets['aset_no_name1'].values())
        data2 = next(co.arraysets['aset_no_name2'].values())
        assert np.allclose(data1, randomsizedarray)
        assert np.allclose(data2, randomsizedarray / 255)
        co.close()

    def test_writer_arrayset_properties_are_correct(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        d = co.arraysets['_aset']
        assert d.name == '_aset'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.variable_shape is False
        assert d.named_samples is True
        assert d.iswriteable is True
        co.close()

    def test_reader_arrayset_properties_are_correct(self, written_repo, array5by7):
        co = written_repo.checkout(write=False)
        d = co.arraysets['_aset']
        assert d.name == '_aset'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.variable_shape is False
        assert d.named_samples is True
        assert d.iswriteable is False


class TestVariableSizedArrayset(object):

    @pytest.mark.parametrize(
        'test_shapes,shape',
        [[[(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10)],
         [[(10,), (1,), (5,)], (10,)],
         [[(100, 100, 100), (100, 100, 1), (100, 1, 100), (1, 100, 100), (1, 1, 1), (34, 6, 3)], (100, 100, 100)]])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_writer_can_create_variable_size_arrayset(self, written_repo, dtype, test_shapes, shape, backend):
        repo = written_repo
        wco = repo.checkout(write=True)
        wco.arraysets.init_arrayset('varaset', shape=shape, dtype=dtype, variable_shape=True, backend=backend)
        d = wco.arraysets['varaset']

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
    def test_reader_recieves_expected_values_for_variable_size_arrayset(self, written_repo, dtype, test_shapes, shape, backend):
        repo = written_repo
        wco = repo.checkout(write=True)
        wco.arraysets.init_arrayset('varaset', shape=shape, dtype=dtype, variable_shape=True, backend=backend)
        wd = wco.arraysets['varaset']

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
        rd = rco.arraysets['varaset']

        for k, v in arrdict.items():
            # make sure they can work after commit
            assert np.allclose(wd[k], v)
            assert np.allclose(rd[k], v)
        wco.close()
        rco.close()


    @pytest.mark.parametrize('aset_specs', [
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.uint8],
         ['aset2', [(10,), (1,), (5,)], (10,), '00', np.uint8]
         ],
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.uint8],
         ['aset2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.uint8],
         ['aset2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.uint8],
         ['aset2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.float32],
         ['aset2', [(10,), (1,), (5,)], (10,), '00', np.float32]
         ],
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.float32],
         ['aset2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ],
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.float32],
         ['aset2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ],
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.float32],
         ['aset2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ]])
    def test_writer_reader_can_create_read_multiple_variable_size_arrayset(self, written_repo, aset_specs):
        repo = written_repo
        wco = repo.checkout(write=True)
        arrdict = {}
        for aset_spec in aset_specs:
            aset_name, test_shapes, max_shape, backend, dtype = aset_spec
            wco.arraysets.init_arrayset(aset_name, shape=max_shape, dtype=dtype, variable_shape=True, backend=backend)

            arrdict[aset_name] = {}
            for idx, shape in enumerate(test_shapes):
                arr = (np.random.random_sample(shape) * 10).astype(dtype)
                arrdict[aset_name][str(idx)] = arr
                wco.arraysets[aset_name][str(idx)] = arr

        for aset_k in arrdict.keys():
            for samp_k, v in arrdict[aset_k].items():
                # make sure they are good before committed
                assert np.allclose(wco.arraysets[aset_k][samp_k], v)

        wco.commit('first')
        rco = repo.checkout()

        for aset_k in arrdict.keys():
            for samp_k, v in arrdict[aset_k].items():
                # make sure they are good before committed
                assert np.allclose(wco.arraysets[aset_k][samp_k], v)
                assert np.allclose(rco.arraysets[aset_k][samp_k], v)
        wco.close()
        rco.close()

    def test_writer_arrayset_properties_are_correct(self, variable_shape_written_repo):
        co = variable_shape_written_repo.checkout(write=True)
        d = co.arraysets['_aset']
        assert d.name == '_aset'
        assert d.dtype == np.float64
        assert np.allclose(d.shape, (10, 10))
        assert d.variable_shape is True
        assert d.named_samples is True
        assert d.iswriteable is True
        co.close()

    def test_reader_arrayset_properties_are_correct(self, variable_shape_written_repo):
        co = variable_shape_written_repo.checkout(write=False)
        d = co.arraysets['_aset']
        assert d.name == '_aset'
        assert d.dtype == np.float64
        assert np.allclose(d.shape, (10, 10))
        assert d.variable_shape is True
        assert d.named_samples is True
        assert d.iswriteable is False
        co.close()


class TestMultiprocessArraysetReads(object):

    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_external_multi_process_pool(self, repo, backend):
        from multiprocessing import get_context

        masterCmtList = []
        co = repo.checkout(write=True)
        co.arraysets.init_arrayset(name='_aset', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        for cIdx in range(2):
            if cIdx != 0:
                co = repo.checkout(write=True)
            with co.arraysets['_aset'] as d:
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
            ds = nco.arraysets['_aset']
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
        co.arraysets.init_arrayset(name='_aset', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        for cIdx in range(2):
            if cIdx != 0:
                co = repo.checkout(write=True)
            with co.arraysets['_aset'] as d:
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
            ds = nco.arraysets['_aset']
            keys = [str(i) for i in range(20 + (20*cmtIdx))]
            cmtData = ds.get_batch(keys, n_cpus=2)
            for data, sampData in zip(cmtData, sampList):
                assert np.allclose(data, sampData) is True
            cmtIdx += 1
            nco.close()

    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_batch_get_fails_on_superset_of_keys_and_succeeds_on_subset(self, repo, backend):
        co = repo.checkout(write=True)
        co.arraysets.init_arrayset(name='_aset', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        with co.arraysets['_aset'] as d:
            for sIdx in range(20):
                arr = np.random.randn(20, 20).astype(np.float32) * 100
                sName = str(sIdx)
                d[sName] = arr
                masterSampList.append(arr)
            assert d._backend == backend
        cmt = co.commit(f'master commit number one')
        co.close()

        nco = repo.checkout(write=False, commit=cmt)
        ds = nco.arraysets['_aset']

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
        aset = co.arraysets['_aset']
        klist = []
        with aset as ds:
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
        aset = co.arraysets['_aset']
        vlist = []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with aset as ds:
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
        aset = co.arraysets['_aset']
        vlist, klist = [], []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with aset as ds:
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
        aset = co.arraysets['_aset']
        vlist, klist = [], []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with aset as ds:
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
