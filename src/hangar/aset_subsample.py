from contextlib import contextmanager
from typing import List, Tuple, Union, NamedTuple, Sequence, Dict, Iterable

import numpy as np

from .utils import is_suitable_user_key
from .backends import backend_decoder
from .records.hashmachine import array_hash_digest
from .records.parsing import (
    data_record_db_key_from_raw_key,
    data_record_db_val_from_raw_val,
    hash_data_db_key_from_raw_key,
)

KeyType = Union[str, int]
EllipsisType = type(Ellipsis)


class CompatibleArray(NamedTuple):
    compatible: bool
    reason: str


class SubsampleReader(object):

    __slots__ = ('_asetn', '_samplen', '_be_fs', '_specs', '_dflt_backend',
                 '_mode', '__weakref__')

    def __init__(
        self,
        asetn,
        samplen,
        be_handles,
        specs,
        dflt_backend,
        *args, **kwargs
    ):
        self._asetn = asetn
        self._samplen = samplen
        self._be_fs = be_handles
        self._specs = specs
        self._dflt_backend = dflt_backend
        self._mode = 'r'

    def __repr__(self):
        res = f'{self.__class__}('\
              f'aset_name={self._asetn}, '\
              f'sample_name={self._samplen}, '\
              f'default_backend={self._dflt_backend})'
        return res

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__} \
                \n    Arrayset Name            : {self._asetn}\
                \n    Sample Name              : {self._samplen}\
                \n    Mode (read/write)        : {self._mode}\
                \n    Default Backend          : {self._dflt_backend}\
                \n    Number of Subsamples     : {self.__len__()}\n'
        p.text(res)

    def __len__(self):
        return len(self._specs)

    def __contains__(self, key):
        return key in self._specs

    def __getitem__(self, key):
        return self.get(key)

    @property
    def sample_name(self) -> KeyType:
        return self._samplen

    @property
    def arrayset_name(self) -> str:
        return self._asetn

    @property
    def data(self) -> Dict[KeyType, np.ndarray]:
        return dict(self.items())

    def keys(self) -> Iterable[KeyType]:
        for k in self._specs.keys():
            yield k

    def values(self) -> Iterable[np.ndarray]:
        for k in self._specs.keys():
            yield self.get(k)

    def items(self) -> Iterable[Tuple[KeyType, np.ndarray]]:
        for k in self._specs.keys():
            yield (k, self.get(k))

    def get(self, keys: Union[KeyType, Sequence[KeyType], EllipsisType, slice]
            ) -> Union[np.ndarray, Dict[KeyType, np.ndarray]]:

        if isinstance(keys, (str, int)):
            # select subsample(s) with regular keys
            spec = self._specs[keys]
            return self._be_fs[spec.backend].read_data(spec)

        elif isinstance(keys, (list, tuple)):
            # select subsample(s) in sequence ie. [sub1, sub2, ..., subN]]
            res = {}
            for subsample in keys:
                spec = self._specs[subsample]
                res[subsample] = self._be_fs[spec.backend].read_data(spec)
            return res

        elif keys is Ellipsis:
            # select all subsamples
            res = {}
            for subsample, spec in self._specs.items():
                res[subsample] = self._be_fs[spec.backend].read_data(spec)
            return res

        elif isinstance(keys, slice):
            # slice subsamples by sorted order of keys
            res = {}
            subsample_spec_slice = tuple(self._specs.items())[keys]
            for subsample, spec in subsample_spec_slice:
                res[subsample] = self._be_fs[spec.backend].read_data(spec)
            return res

        else:
            raise ValueError(f'subsample keys argument: {keys} not valid format')


class SubsampleWriter(SubsampleReader):

    __slots__ = ('_txn_ctx',)

    def __init__(self, aset_txn_ctx, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mode = 'a'
        self._txn_ctx = aset_txn_ctx

    def __setitem__(self, key, value):
        return self.update(other={key: value})

    def __delitem__(self, keys):
        return self.pop(keys)

    def _verify_array_compatible(self, data):
        return CompatibleArray(True, '')

    @contextmanager
    def _txn_be_fs_context(self):
        try:
            if self._txn_ctx.is_conman:
                tmpconmap = False
                yield self._txn_ctx
            else:
                tmpconmap = True
                self._txn_ctx = self._txn_ctx.open_write()
                for k in self._be_fs.keys():
                    self._be_fs[k].__enter__()
                yield self._txn_ctx
        finally:
            if tmpconmap:
                for k in self._be_fs.keys():
                    self._be_fs[k].__exit__()
                self._txn_ctx.close_write()

    def update(
        self,
        other: Union[Dict[KeyType, np.ndarray],
                     Tuple[KeyType, np.ndarray],
                     List[Union[KeyType, np.ndarray]],
                     Sequence[Union[
                         Tuple[KeyType, np.ndarray],
                         List[Union[KeyType, np.ndarray]]]]],
    ) -> Tuple[KeyType]:

        # ------------------------ argument checking --------------------------

        try:
            if isinstance(other, dict):
                for k, v in other.items():
                    if not is_suitable_user_key(k):
                        raise ValueError(f'dict key: {k} not suitable name')
                    isCompat = self._verify_array_compatible(v)
                    if not isCompat.compatible:
                        raise ValueError(isCompat.reason)
                data_map = other

            elif isinstance(other, (list, tuple)):
                if isinstance(other[0], (list, tuple)):
                    for item_data in other:
                        if len(item_data) != 2:
                            raise ValueError(
                                f'container for subsample name/data must be length 2. '
                                f'not {len(item_data)} for {item_data}')
                        subsamplen, data = item_data
                        if not is_suitable_user_key(subsamplen):
                            raise ValueError(f'data sample name: {subsamplen} not suitable.')
                        isCompat = self._verify_array_compatible(data)
                        if not isCompat.compatible:
                            raise ValueError(isCompat.reason)
                    data_map = dict(list(other))
                else:
                    if len(other) != 2:
                        raise ValueError(
                            f'container for subsample name/data must be length 2. '
                            f'not {len(other)} for {other}')
                    subsamplen, data = other
                    if not is_suitable_user_key(subsamplen):
                        raise ValueError(f'data sample name: {subsamplen} not suitable.')
                    isCompat = self._verify_array_compatible(data)
                    if not isCompat.compatible:
                        raise ValueError(isCompat.reason)
                    data_map = dict([other])

        except ValueError as e:
            raise e from None

        # ------------------- add data to storage backend ---------------------

        with self._txn_be_fs_context() as ctx:
            for subsamplen, data in data_map.items():
                full_hash = array_hash_digest(data)
                hashKey = hash_data_db_key_from_raw_key(full_hash)
                # check if data record already exists with given key
                dataRecKey = data_record_db_key_from_raw_key(
                    aset_name=self._asetn,
                    data_name=self._samplen,
                    subsample=subsamplen)

                # write new data if data hash does not exist
                existingHashVal = ctx.hashTxn.get(hashKey, default=False)
                if existingHashVal is False:
                    hashVal = self._be_fs[self._dflt_backend].write_data(data)
                    ctx.hashTxn.put(hashKey, hashVal)
                    ctx.stageHashTxn.put(hashKey, hashVal)
                    hash_spec = backend_decoder(hashVal)
                else:
                    hash_spec = backend_decoder(existingHashVal)
                self._specs.update([(subsamplen, hash_spec)])

                # add the record to the db
                dataRecVal = data_record_db_val_from_raw_val(full_hash)
                ctx.dataTxn.put(dataRecKey, dataRecVal)

        return tuple(data_map.keys())

    def pop(self, keys: Union[KeyType, List[KeyType], Tuple[KeyType]]
            ) -> Union[KeyType, Tuple[KeyType]]:

        # ------------------------ argument checking --------------------------

        if isinstance(keys, (list, tuple)):
            if not all([k in self._specs for k in keys]):
                raise KeyError(f'All subsample keys: {keys} must exist in sample.')
            subsamples = keys
        elif isinstance(keys, (str, int)):
            if keys not in self._specs:
                raise KeyError(f'Subsample key: {keys} must exist in sample.')
            subsamples = [keys]
        else:
            raise TypeError(f'Invalid argument type of `keys` param: {type(keys)}')

        # ------------------------- db modification ---------------------------

        with self._txn_be_fs_context() as ctx:
            for subsample in subsamples:
                dbKey = data_record_db_key_from_raw_key(self._asetn,
                                                        self._samplen,
                                                        subsample=subsample)
                isRecordDeleted = ctx.dataTxn.delete(dbKey)
                if isRecordDeleted is False:
                    raise KeyError(
                        f'Subsample key {subsample} does not exist in sample '
                        f'{self._samplen} of arrayset {self._asetn}')
                del self._specs[subsample]
        return keys