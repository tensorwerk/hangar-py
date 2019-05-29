import re
from collections import namedtuple

from .. import config

SEP = config.get('hangar.seps.key')
LISTSEP = config.get('hangar.seps.list')
SLICESEP = config.get('hangar.seps.slice')
HASHSEP = config.get('hangar.seps.hash')

# -----------------------------------------------------------------------------

# match and remove the following characters: '['   ']'   '('   ')'   ','
# DataShapeReplacementRegEx = re.compile('[,\(\)\[\]]')


# -----------------------------------------------------------------------------


class HDF5_00_Parser(object):

    __slots__ = ['FmtCode', 'FmtCodeIdx', 'DataShapeReplacementRE', 'DataHashSpec']

    def __init__(self):

        self.FmtCode = f'00{SEP}'
        self.FmtCodeIdx = len(self.FmtCode)

        # match and remove the following characters: '['   ']'   '('   ')'   ','
        self.DataShapeReplacementRE = re.compile('[,\(\)\[\]]')

        self.DataHashSpec = namedtuple(
            typename='DataHashVal',
            field_names=['schema', 'instance', 'dataset', 'dataset_idx', 'shape'])

    def encode(self, schema, instance, dataset, dataset_idx, shape) -> bytes:
        '''converts the hdf5 data has spec to an appropriate db value

        .. Note:: This is the inverse of `hash_data_raw_val_from_db_val()`. Any changes in
                db value format must be reflected in both functions.

        Parameters
        ----------
        schema : str
            hdf5 schema hash to find this data piece in.
        instance : str
            file name (schema instance) of the hdf5 file to find this data piece in.
        dataset : str
            collection (ie. hdf5 dataset) name to find find this data piece.
        dataset_idx : int or str
            collection first axis index in which this data piece resides.
        shape : tuple
            shape of the data sample written to the collection idx. ie:
            what subslices of the hdf5 dataset should be read to retrieve
            the sample as recorded.

        Returns
        -------
        bytes
            hash data db value recording all input specifications.
        '''
        out_str = f'{self.FmtCode}'\
                  f'{schema}{LISTSEP}{instance}'\
                  f'{HASHSEP}'\
                  f'{dataset}{LISTSEP}{dataset_idx}'\
                  f'{SLICESEP}'\
                  f'{self.DataShapeReplacementRE.sub("", str(shape))}'
        return out_str.encode()

    def decode(self, db_val: bytes) -> namedtuple:
        '''converts an hdf5 data hash db val into an hdf5 data python spec.

        .. Note:: This is the inverse of `hash_data_db_val_from_raw_val()`. Any changes in
                db value format must be reflected in both functions.

        Parameters
        ----------
        db_val : bytestring
            data hash db value

        Returns
        -------
        namedtuple
            hdf5 data hash specification in DataHashVal named tuple format
        '''
        db_str = db_val.decode()[self.FmtCodeIdx:]

        schema_vals, _, dset_vals = db_str.partition(HASHSEP)
        schema, instance = schema_vals.split(LISTSEP)

        dataset_vs, _, shape_vs = dset_vals.rpartition(SLICESEP)
        dataset, dataset_idx = dataset_vs.split(LISTSEP)
        # if the data is of empty shape -> ()
        shape = () if shape_vs == '' else tuple([int(x) for x in shape_vs.split(LISTSEP)])

        raw_val = self.DataHashSpec(schema=schema,
                                    instance=instance,
                                    dataset=dataset,
                                    dataset_idx=dataset_idx,
                                    shape=shape)
        return raw_val


CODE_BACKEND_MAP = {
    # LOCALS -> [0:50]
    b'00': HDF5_00_Parser(),  # hdf5_00
    b'01': None,              # numpy_00
    b'02': None,              # tiledb_00
    # REMOTES -> [50:100]
    b'50': None,              # remote_00
    b'51': None,              # url_00
}

BACKEND_CODE_MAP = {
    # LOCALS -> [0:50]
    'hdf5_00': HDF5_00_Parser(),  # 00
    'numpy_00': None,             # 01
    'tiledb_00': None,            # 02
    # REMOTES -> [50:100]
    'remote_00': None,            # 50
    'url_00': None,               # 51
}

LOCATION_CODE_BACKEND_MAP = {
    # LOCALS -> [0:50]
    '00': 'hdf5_00',
    '01': 'numpy_00',
    '02': 'tiledb_00',
    # REMOTES -> [50:100]
    '50': 'remote_00',
    '51': 'url_00',
}
LOCATION_BACKEND_CODE_MAP = dict([[v, k] for k, v in LOCATION_CODE_BACKEND_MAP.items()])


def backend_decoder(db_val: bytes) -> namedtuple:

    parser = CODE_BACKEND_MAP[db_val[:2]]
    decoded = parser.decode(db_val)
    return decoded


def backend_decoder_name(db_val: bytes) -> str:
    val = db_val.decode()[:2]
    name = LOCATION_CODE_BACKEND_MAP[val]
    return name


def backend_encoder(backend, *args, **kwargs):

    parser = BACKEND_CODE_MAP[backend]
    encoded = parser.encode(*args, **kwargs)
    return encoded