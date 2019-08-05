import pytest
import numpy as np


class TestDatacell(object):

    def test_invalid_dcellname(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        with pytest.raises(ValueError):
            co.datacells.init_datacell(name='invalid name', prototype=randomsizedarray)
        co.close()

    def test_read_only_mode(self, written_repo):
        import hangar
        co = written_repo.checkout()
        assert isinstance(co, hangar.checkout.ReaderCheckout)
        assert co.datacells.init_datacell is None
        assert co.datacells.remove_dcell is None
        assert len(co.datacells['_dcell']) == 0
        co.close()

    def test_get_datacell(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)

        # getting the datacell with `get`
        dcellOld = co.datacells.get('_dcell')
        dcellOldPath = dcellOld._path
        dcellOldDcelln = dcellOld._dcelln
        dcellOldDefaultSchemaHash = dcellOld._default_schema_hash

        dcellOld.add(array5by7, '1')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()

        # getting datacell with dictionary like style method
        dcellNew = co.datacells['_dcell']

        assert np.allclose(dcellNew['1'], array5by7)
        assert dcellOldPath == dcellNew._path
        assert dcellOldDcelln == dcellNew._dcelln
        assert dcellOldDefaultSchemaHash == dcellNew._default_schema_hash
        co.close()

    @pytest.mark.parametrize("dcell_backend", ['00', '10'])
    def test_remove_datacell(self, dcell_backend, written_repo):
        co = written_repo.checkout(write=True)
        co.datacells.remove_dcell('_dcell')
        with pytest.raises(KeyError):
            co.datacells.remove_dcell('_dcell')

        co.datacells.init_datacell(name='_dcell', shape=(5, 7), dtype=np.float64, backend=dcell_backend)
        assert len(co.datacells) == 1
        co.datacells.remove_dcell('_dcell')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        assert len(co.datacells) == 0

        co.datacells.init_datacell(name='_dcell', shape=(5, 7), dtype=np.float64, backend=dcell_backend)
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout(write=True)
        assert len(co.datacells) == 1
        del co.datacells['_dcell']
        assert len(co.datacells) == 0
        co.commit('this is a commit message')
        co.close()

    @pytest.mark.parametrize("dcell_backend", ['00', '10'])
    def test_init_again(self, dcell_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.datacells.init_datacell('dcell', prototype=randomsizedarray, backend=dcell_backend)
        with pytest.raises(LookupError):
            co.datacells.init_datacell('dcell', prototype=randomsizedarray, backend=dcell_backend)
        co.close()

    @pytest.mark.parametrize("dcell_backend", ['00', '10'])
    def test_datacell_with_more_dimension(self, dcell_backend, repo):
        co = repo.checkout(write=True)
        shape = (0, 1, 2)
        with pytest.raises(ValueError):
            co.datacells.init_datacell('dcell', shape=shape, dtype=np.int, backend=dcell_backend)
        shape = [1] * 31
        dcell = co.datacells.init_datacell('dcell1', shape=shape, dtype=np.int, backend=dcell_backend)
        assert len(dcell._schema_max_shape) == 31
        shape = [1] * 32
        with pytest.raises(ValueError):
            # maximum tensor rank must be <= 31
            co.datacells.init_datacell('dcell2', shape=shape, dtype=np.int, backend=dcell_backend)
        co.close()

    @pytest.mark.parametrize("dcell_backend", ['00', '10'])
    def test_datacell_with_empty_dimension(self, dcell_backend, repo):
        co = repo.checkout(write=True)
        arr = np.array(1, dtype=np.int64)
        dcell = co.datacells.init_datacell('dcell1', shape=(), dtype=np.int64, backend=dcell_backend)
        dcell['1'] = arr
        co.commit('this is a commit message')
        dcell = co.datacells.init_datacell('dcell2', prototype=arr)
        dcell['1'] = arr
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        dcell1 = co.datacells['dcell1']
        dcell2 = co.datacells['dcell2']
        assert np.allclose(dcell1['1'], arr)
        assert np.allclose(dcell2['1'], arr)
        co.close()

    @pytest.mark.parametrize("dcell_backend", ['00', '10'])
    def test_datacell_with_int_specifier_as_dimension(self, dcell_backend, repo):
        co = repo.checkout(write=True)
        arr = np.arange(10, dtype=np.int64)
        dcell = co.datacells.init_datacell('dcell1', shape=10, dtype=np.int64, backend=dcell_backend)
        dcell['1'] = arr
        co.commit('this is a commit message')
        arr2 = np.array(53, dtype=np.int64)
        dcell = co.datacells.init_datacell('dcell2', prototype=arr2)
        dcell['1'] = arr2
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        dcell1 = co.datacells['dcell1']
        dcell2 = co.datacells['dcell2']
        assert np.allclose(dcell1['1'], arr)
        assert np.allclose(dcell2['1'], arr2)
        co.close()


class TestDataWithFixedSizedDatacell(object):

    @pytest.mark.parametrize("dcell1_backend", ['00', '10'])
    @pytest.mark.parametrize("dcell2_backend", ['00', '10'])
    @pytest.mark.parametrize("dcell3_backend", ['00', '10'])
    def test_iterating_over(self, dcell1_backend, dcell2_backend, dcell3_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        all_tensors = []
        dcell1 = co.datacells.init_datacell('dcell1', prototype=randomsizedarray, backend=dcell1_backend)
        dcell2 = co.datacells.init_datacell('dcell2', shape=(2, 2), dtype=np.int, backend=dcell2_backend)
        dcell3 = co.datacells.init_datacell('dcell3', shape=(3, 4), dtype=np.float32, backend=dcell3_backend)

        with dcell1 as d1, dcell2 as d2, dcell3 as d3:
            d1['1'] = randomsizedarray
            d1['2'] = np.zeros_like(randomsizedarray)
            d1['3'] = np.zeros_like(randomsizedarray) + 5

            d2['1'] = np.ones((2, 2), dtype=np.int)
            d2['2'] = np.ones((2, 2), dtype=np.int) * 5
            d2['3'] = np.zeros((2, 2), dtype=np.int)

            d3['1'] = np.ones((3, 4), dtype=np.float32)
            d3['2'] = np.ones((3, 4), dtype=np.float32) * 7
            d3['3'] = np.zeros((3, 4), dtype=np.float32)

        all_tensors.extend([dcell1['1'], dcell1['2'], dcell1['3']])
        all_tensors.extend([dcell2['1'], dcell2['2'], dcell2['3']])
        all_tensors.extend([dcell3['1'], dcell3['2'], dcell3['3']])

        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        # iterating over .items()
        tensors_in_the_order = iter(all_tensors)
        for dname, dcell in co.datacells.items():
            assert dcell._dcelln == dname
            for sname, sample in dcell.items():
                assert np.allclose(sample, next(tensors_in_the_order))

        # iterating over .keys()
        tensors_in_the_order = iter(all_tensors)
        for dname in co.datacells.keys():
            for sname in co.datacells[dname].keys():
                assert np.allclose(co.datacells[dname][sname], next(tensors_in_the_order))

        # iterating over .values()
        tensors_in_the_order = iter(all_tensors)
        for dcell in co.datacells.values():
            for sample in dcell.values():
                assert np.allclose(sample, next(tensors_in_the_order))
        co.close()

    def test_get_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datacells['_dcell']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.datacells['_dcell']['1'], co.datacells.get('_dcell').get('1'), array5by7)
        co.close()

    def test_add_data_str_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dcell = co.datacells['_dcell']
        with pytest.raises(KeyError):
            dcell['somerandomkey']

        dcell['1'] = array5by7
        dcell.add(array5by7, '2')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.datacells['_dcell']['1'], array5by7)
        assert np.allclose(co.datacells['_dcell']['2'], array5by7)
        co.close()

    def test_add_data_int_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dcell = co.datacells['_dcell']

        dcell[1] = array5by7
        secondArray = array5by7 + 1
        dcell.add(secondArray, 2)
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.datacells['_dcell'][1], array5by7)
        assert np.allclose(co.datacells['_dcell'][2], secondArray)
        co.close()

    def test_cannot_add_data_negative_int_key(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dcell = co.datacells['_dcell']
        with pytest.raises(ValueError):
            dcell[-1] = array5by7
        assert len(co.datacells['_dcell']) == 0
        co.close()

    def test_cannot_add_data_float_key(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dcell = co.datacells['_dcell']
        with pytest.raises(ValueError):
            dcell[2.1] = array5by7
        with pytest.raises(ValueError):
            dcell[0.0] = array5by7
        assert len(co.datacells['_dcell']) == 0
        co.close()

    def test_add_data_mixed_int_str_keys(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dcell = co.datacells['_dcell']

        dcell[1] = array5by7
        newFirstArray = array5by7 + 1
        dcell['1'] = newFirstArray
        secondArray = array5by7 + 2
        dcell.add(secondArray, 2)
        thirdArray = array5by7 + 3
        dcell.add(thirdArray, '2')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.datacells['_dcell'][1], array5by7)
        assert np.allclose(co.datacells['_dcell']['1'], newFirstArray)
        assert np.allclose(co.datacells['_dcell'][2], secondArray)
        assert np.allclose(co.datacells['_dcell']['2'], thirdArray)
        co.close()

    def test_add_with_wrong_argument_order(self, w_checkout, array5by7):
        dcell = w_checkout.datacells['_dcell']
        with pytest.raises(ValueError):
            dcell.add('1', array5by7)

    def test_multiple_data_single_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datacells['_dcell'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datacells['_dcell']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        dcell = co.datacells['_dcell']
        assert len(dcell) == 2
        assert list(dcell.keys()) == ['1', '2']
        assert np.allclose(dcell['1'], array5by7)
        co.close()

    def test_multiple_data_multiple_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datacells['_dcell'].add(array5by7, '1')
        co.commit('this is a commit message')
        new_array = np.zeros_like(array5by7)
        co.datacells['_dcell']['2'] = new_array
        co.close()

        new_new_array = new_array + 5
        co = written_repo.checkout(write=True)
        co.datacells['_dcell']['3'] = new_new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        dcell = co.datacells['_dcell']
        assert np.allclose(dcell['1'], array5by7)
        assert np.allclose(dcell['2'], new_array)
        assert np.allclose(dcell['3'], new_new_array)
        co.close()

    def test_added_but_not_commited(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datacells['_dcell'].add(array5by7, '1')
        co.close()

        with pytest.raises(PermissionError):
            co.commit('this is a commit message')

        co = written_repo.checkout()
        dcell = co.datacells['_dcell']
        with pytest.raises(KeyError):
            dcell['1']
        co.close()

    def test_remove_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datacells['_dcell'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datacells['_dcell']['2'] = new_array
        co.datacells['_dcell']['3'] = new_array + 5
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        co.datacells['_dcell'].remove('1')
        del co.datacells['_dcell']['3']
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        with pytest.raises(KeyError):
            co.datacells['_dcell']['1']
        with pytest.raises(KeyError):
            co.datacells['_dcell']['3']
        assert len(co.datacells['_dcell']) == 1
        assert np.allclose(co.datacells['_dcell']['2'], new_array)
        co.close()

    def test_remove_all_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datacells['_dcell'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datacells['_dcell']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        co.datacells['_dcell'].remove('1')
        co.datacells['_dcell'].remove('2')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        with pytest.raises(KeyError):
            # removal of all data removes the datacell
            co.datacells['_dcell']
        co.close()

        # recreating same and verifying
        co = written_repo.checkout(write=True)
        co.datacells.init_datacell('_dcell', prototype=array5by7)
        co.datacells['_dcell']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.datacells['_dcell']['1'], array5by7)
        co.close()

    @pytest.mark.parametrize("dcell1_backend", ['00', '10'])
    @pytest.mark.parametrize("dcell2_backend", ['00', '10'])
    def test_multiple_datacells_single_commit(self, dcell1_backend, dcell2_backend, written_repo, randomsizedarray):
        co = written_repo.checkout(write=True)
        dcell1 = co.datacells.init_datacell('dcell1', prototype=randomsizedarray, backend=dcell1_backend)
        dcell2 = co.datacells.init_datacell('dcell2', prototype=randomsizedarray, backend=dcell2_backend)
        dcell1['arr'] = randomsizedarray
        dcell2['arr'] = randomsizedarray
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.datacells['dcell1']['arr'], randomsizedarray)
        assert np.allclose(co.datacells['dcell2']['arr'], randomsizedarray)
        co.close()

    @pytest.mark.parametrize("dcell1_backend", ['00', '10'])
    @pytest.mark.parametrize("dcell2_backend", ['00', '10'])
    def test_prototype_and_shape(self, dcell1_backend, dcell2_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dcell1 = co.datacells.init_datacell(
            'dcell1', prototype=randomsizedarray, backend=dcell1_backend)
        dcell2 = co.datacells.init_datacell(
            'dcell2', shape=randomsizedarray.shape, dtype=randomsizedarray.dtype, backend=dcell2_backend)

        newarray = np.random.random(randomsizedarray.shape).astype(randomsizedarray.dtype)
        dcell1['arr1'] = newarray
        dcell2['arr'] = newarray
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert np.allclose(co.datacells['dcell1']['arr1'], newarray)
        assert np.allclose(co.datacells['dcell2']['arr'], newarray)
        co.close()

    def test_samples_without_name(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dcell = co.datacells.init_datacell('dcell', prototype=randomsizedarray)
        with pytest.raises(ValueError):
            dcell.add(randomsizedarray)

        dcell_no_name = co.datacells.init_datacell('dcell_no_name',
                                                prototype=randomsizedarray,
                                                named_samples=False)
        dcell_no_name.add(randomsizedarray)
        assert np.allclose(next(dcell_no_name.values()), randomsizedarray)
        co.close()

    def test_different_data_types_and_shapes(self, repo):
        co = repo.checkout(write=True)
        shape = (2, 3)
        dtype = np.int
        another_dtype = np.float64
        another_shape = (3, 4)
        arr = np.random.random(shape).astype(dtype)
        dcell = co.datacells.init_datacell('dcell', shape=shape, dtype=dtype)
        dcell['1'] = arr

        newarr = np.random.random(shape).astype(another_dtype)
        with pytest.raises(ValueError):
            dcell['2'] = newarr

        newarr = np.random.random(another_shape).astype(dtype)
        with pytest.raises(ValueError):
            dcell['3'] = newarr
        co.close()

    @pytest.mark.parametrize("dcell_backend", ['00', '10'])
    def test_adding_same_data_again_with_same_name(self, dcell_backend, repo, array5by7):
        co = repo.checkout(write=True)
        dcell = co.datacells.init_datacell('dcell', prototype=array5by7, backend=dcell_backend)
        dcell['1'] = array5by7
        with pytest.raises(LookupError):
            # raises before commit
            dcell['1'] = array5by7
        co.commit('this is a commit message')
        with pytest.raises(LookupError):
            # raises after commit
            dcell['1'] = array5by7
        co.close()
        co = repo.checkout(write=True)
        dcell = co.datacells['dcell']
        with pytest.raises(LookupError):
            # raises in another checkout
            dcell['1'] = array5by7
        co.close()

    @pytest.mark.parametrize("dcell_backend", ['00', '10'])
    def test_writer_context_manager_datacell_add_sample(self, dcell_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dcell = co.datacells.init_datacell('dcell', prototype=randomsizedarray, backend=dcell_backend)
        with co.datacells['dcell'] as dcell:
            dcell.add(randomsizedarray, '1')
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        assert np.allclose(co.datacells['dcell']['1'], randomsizedarray)
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

    @pytest.mark.parametrize("dcell_backend", ['00', '10'])
    def test_datacell_context_manager_dcell_sample_and_metadata_add(self, dcell_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dcell = co.datacells.init_datacell('dcell', prototype=randomsizedarray, backend=dcell_backend)
        with co.datacells['dcell'] as dcell:
            dcell.add(randomsizedarray, '1')
            co.metadata['hello'] = 'world'
        with co.metadata as metadata:
            newarr = randomsizedarray + 1
            dcell['2'] = newarr
            metadata.add('key', 'val')
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert np.allclose(co.datacells['dcell']['1'], randomsizedarray)
        assert np.allclose(co.datacells['dcell'].get('2'), newarr)
        assert co.metadata['key'] == 'val'
        assert co.metadata.get('hello') == 'world'
        co.close()

    @pytest.mark.parametrize("dcell1_backend", ['00', '10'])
    @pytest.mark.parametrize("dcell2_backend", ['00', '10'])
    def test_bulk_add(self, dcell1_backend, dcell2_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.datacells.init_datacell(
            'dcell_no_name1',
            prototype=randomsizedarray,
            named_samples=False,
            backend=dcell1_backend)
        co.datacells.init_datacell(
            'dcell_no_name2',
            prototype=randomsizedarray,
            named_samples=False,
            backend=dcell2_backend)
        co.commit('this is a commit message')

        # dummy additino with wrong key
        with pytest.raises(KeyError):
            co.datacells.multi_add(
                {
                    'dcell_no_name2': randomsizedarray / 255,
                    'dummykey': randomsizedarray
                })
        # making sure above addition did not add partial data
        with pytest.raises(RuntimeError):
            co.commit('this is a commit message')

        # proper addition and verification
        co.datacells.multi_add(
            {
                'dcell_no_name1': randomsizedarray,
                'dcell_no_name2': randomsizedarray / 255
            })
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        data1 = next(co.datacells['dcell_no_name1'].values())
        data2 = next(co.datacells['dcell_no_name2'].values())
        assert np.allclose(data1, randomsizedarray)
        assert np.allclose(data2, randomsizedarray / 255)
        co.close()

    def test_writer_datacell_properties_are_correct(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        d = co.datacells['_dcell']
        assert d.name == '_dcell'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.variable_shape is False
        assert d.named_samples is True
        assert d.iswriteable is True
        co.close()

    def test_reader_datacell_properties_are_correct(self, written_repo, array5by7):
        co = written_repo.checkout(write=False)
        d = co.datacells['_dcell']
        assert d.name == '_dcell'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.variable_shape is False
        assert d.named_samples is True
        assert d.iswriteable is False


class TestVariableSizedDatacell(object):

    @pytest.mark.parametrize(
        'test_shapes,shape',
        [[[(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10)],
         [[(10,), (1,), (5,)], (10,)],
         [[(100, 100, 100), (100, 100, 1), (100, 1, 100), (1, 100, 100), (1, 1, 1), (34, 6, 3)], (100, 100, 100)]])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_writer_can_create_variable_size_datacell(self, written_repo, dtype, test_shapes, shape, backend):
        repo = written_repo
        wco = repo.checkout(write=True)
        wco.datacells.init_datacell('vardcell', shape=shape, dtype=dtype, variable_shape=True, backend=backend)
        d = wco.datacells['vardcell']

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
    def test_reader_recieves_expected_values_for_variable_size_datacell(self, written_repo, dtype, test_shapes, shape, backend):
        repo = written_repo
        wco = repo.checkout(write=True)
        wco.datacells.init_datacell('vardcell', shape=shape, dtype=dtype, variable_shape=True, backend=backend)
        wd = wco.datacells['vardcell']

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
        rd = rco.datacells['vardcell']

        for k, v in arrdict.items():
            # make sure they can work after commit
            assert np.allclose(wd[k], v)
            assert np.allclose(rd[k], v)
        wco.close()
        rco.close()


    @pytest.mark.parametrize('dcell_specs', [
        [['dcell1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.uint8],
         ['dcell2', [(10,), (1,), (5,)], (10,), '00', np.uint8]
         ],
        [['dcell1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.uint8],
         ['dcell2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['dcell1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.uint8],
         ['dcell2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['dcell1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.uint8],
         ['dcell2', [(10,), (1,), (5,)], (10,), '10', np.uint8]
         ],
        [['dcell1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.float32],
         ['dcell2', [(10,), (1,), (5,)], (10,), '00', np.float32]
         ],
        [['dcell1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '00', np.float32],
         ['dcell2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ],
        [['dcell1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.float32],
         ['dcell2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ],
        [['dcell1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10), '10', np.float32],
         ['dcell2', [(10,), (1,), (5,)], (10,), '10', np.float32]
         ]])
    def test_writer_reader_can_create_read_multiple_variable_size_datacell(self, written_repo, dcell_specs):
        repo = written_repo
        wco = repo.checkout(write=True)
        arrdict = {}
        for dcell_spec in dcell_specs:
            dcell_name, test_shapes, max_shape, backend, dtype = dcell_spec
            wco.datacells.init_datacell(dcell_name, shape=max_shape, dtype=dtype, variable_shape=True, backend=backend)

            arrdict[dcell_name] = {}
            for idx, shape in enumerate(test_shapes):
                arr = (np.random.random_sample(shape) * 10).astype(dtype)
                arrdict[dcell_name][str(idx)] = arr
                wco.datacells[dcell_name][str(idx)] = arr

        for dcell_k in arrdict.keys():
            for samp_k, v in arrdict[dcell_k].items():
                # make sure they are good before committed
                assert np.allclose(wco.datacells[dcell_k][samp_k], v)

        wco.commit('first')
        rco = repo.checkout()

        for dcell_k in arrdict.keys():
            for samp_k, v in arrdict[dcell_k].items():
                # make sure they are good before committed
                assert np.allclose(wco.datacells[dcell_k][samp_k], v)
                assert np.allclose(rco.datacells[dcell_k][samp_k], v)
        wco.close()
        rco.close()

    def test_writer_datacell_properties_are_correct(self, variable_shape_written_repo):
        co = variable_shape_written_repo.checkout(write=True)
        d = co.datacells['_dcell']
        assert d.name == '_dcell'
        assert d.dtype == np.float64
        assert np.allclose(d.shape, (10, 10))
        assert d.variable_shape is True
        assert d.named_samples is True
        assert d.iswriteable is True
        co.close()

    def test_reader_datacell_properties_are_correct(self, variable_shape_written_repo):
        co = variable_shape_written_repo.checkout(write=False)
        d = co.datacells['_dcell']
        assert d.name == '_dcell'
        assert d.dtype == np.float64
        assert np.allclose(d.shape, (10, 10))
        assert d.variable_shape is True
        assert d.named_samples is True
        assert d.iswriteable is False
        co.close()


class TestMultiprocessDatacellReads(object):

    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_external_multi_process_pool(self, repo, backend):
        from multiprocessing import get_context

        masterCmtList = []
        co = repo.checkout(write=True)
        co.datacells.init_datacell(name='_dcell', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        for cIdx in range(2):
            if cIdx != 0:
                co = repo.checkout(write=True)
            with co.datacells['_dcell'] as d:
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
            ds = nco.datacells['_dcell']
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
        co.datacells.init_datacell(name='_dcell', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        for cIdx in range(2):
            if cIdx != 0:
                co = repo.checkout(write=True)
            with co.datacells['_dcell'] as d:
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
            ds = nco.datacells['_dcell']
            keys = [str(i) for i in range(20 + (20*cmtIdx))]
            cmtData = ds.get_batch(keys, n_cpus=2)
            for data, sampData in zip(cmtData, sampList):
                assert np.allclose(data, sampData) is True
            cmtIdx += 1
            nco.close()

    @pytest.mark.parametrize('backend', ['00', '10'])
    def test_batch_get_fails_on_superset_of_keys_and_succeeds_on_subset(self, repo, backend):
        co = repo.checkout(write=True)
        co.datacells.init_datacell(name='_dcell', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        with co.datacells['_dcell'] as d:
            for sIdx in range(20):
                arr = np.random.randn(20, 20).astype(np.float32) * 100
                sName = str(sIdx)
                d[sName] = arr
                masterSampList.append(arr)
            assert d._backend == backend
        cmt = co.commit(f'master commit number one')
        co.close()

        nco = repo.checkout(write=False, commit=cmt)
        ds = nco.datacells['_dcell']

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
        dcell = co.datacells['_dcell']
        klist = []
        with dcell as ds:
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
        dcell = co.datacells['_dcell']
        vlist = []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with dcell as ds:
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
        dcell = co.datacells['_dcell']
        vlist, klist = [], []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with dcell as ds:
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
        dcell = co.datacells['_dcell']
        vlist, klist = [], []
        mysample = np.random.randn(5, 7).astype(np.float32)
        with dcell as ds:
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
