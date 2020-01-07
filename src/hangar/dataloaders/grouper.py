import numpy as np

from ..arrayset import ArraysetDataReader
from ..records.hashmachine import array_hash_digest

from collections import defaultdict
from typing import Sequence, Union, Iterable, NamedTuple, Tuple


# -------------------------- typehints ---------------------------------------


ArraysetSampleNames = Sequence[Union[str, int]]

SampleGroup = NamedTuple('SampleGroup', [
    ('group', np.ndarray),
    ('samples', Union[str, int])])


# ------------------------------------------------------------------------------


class FakeNumpyKeyDict(object):
    def __init__(self, group_spec_samples, group_spec_value, group_digest_spec):
        self._group_spec_samples = group_spec_samples
        self._group_spec_value = group_spec_value
        self._group_digest_spec = group_digest_spec

    def __getitem__(self, key: np.ndarray) -> ArraysetSampleNames:
        digest = array_hash_digest(key)
        spec = self._group_digest_spec[digest]
        samples = self._group_spec_samples[spec]
        return samples

    def get(self, key: np.ndarray) -> ArraysetSampleNames:
        return self.__getitem__(key)

    def __setitem__(self, key, val):
        raise PermissionError('Not User Editable')

    def __delitem__(self, key):
        raise PermissionError('Not User Editable')

    def __len__(self) -> int:
        return len(self._group_digest_spec)

    def __contains__(self, key: np.ndarray) -> bool:
        digest = array_hash_digest(key)
        res = True if digest in self._group_digest_spec else False
        return res

    def __iter__(self) -> Iterable[np.ndarray]:
        for spec in self._group_digest_spec.values():
            yield self._group_spec_value[spec]

    def keys(self) -> Iterable[np.ndarray]:
        for spec in self._group_digest_spec.values():
            yield self._group_spec_value[spec]

    def values(self) -> Iterable[ArraysetSampleNames]:
        for spec in self._group_digest_spec.values():
            yield self._group_spec_samples[spec]

    def items(self) -> Iterable[Tuple[np.ndarray, ArraysetSampleNames]]:
        for spec in self._group_digest_spec.values():
            yield (self._group_spec_value[spec], self._group_spec_samples[spec])

    def __repr__(self):
        print('Mapping: Group Data Value -> Sample Name')
        for k, v in self.items():
            print(k, v)

    def _repr_pretty_(self, p, cycle):
        res = f'Mapping: Group Data Value -> Sample Name \n'
        for k, v in self.items():
            res += f'\n {k} :: {v} \n'
        p.text(res)


# ---------------------------- MAIN METHOD ------------------------------------


class GroupedArraysetDataReader(object):
    '''Pass in an arrayset and automatically find sample groups.
    '''

    def __init__(self, arrayset: ArraysetDataReader, *args, **kwargs):

        self.__arrayset = arrayset  # TODO: Do we actually need to keep this around?
        self._group_spec_samples = defaultdict(list)
        self._group_spec_value = {}
        self._group_digest_spec = {}

        self._setup()
        self._group_samples = FakeNumpyKeyDict(
            self._group_spec_samples,
            self._group_spec_value,
            self._group_digest_spec)

    def _setup(self):
        for name, bespec in self.__arrayset._sspecs.items():
            self._group_spec_samples[bespec].append(name)
        for spec, names in self._group_spec_samples.items():
            data = self.__arrayset._fs[spec.backend].read_data(spec)
            self._group_spec_value[spec] = data
            digest = array_hash_digest(data)
            self._group_digest_spec[digest] = spec

    @property
    def groups(self) -> Iterable[np.ndarray]:
        for spec in self._group_digest_spec.values():
            yield self._group_spec_value[spec]

    @property
    def group_samples(self):
        return self._group_samples