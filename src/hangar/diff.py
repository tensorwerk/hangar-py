import logging
from collections import namedtuple
from functools import partial

from .records import parsing

logger = logging.getLogger(__name__)


# --------------------- Differ Primitives ------------------------------------

MetaRecord = namedtuple('MetaRecord',
    field_names=['meta_key', 'meta_hash'])

SamplesDataRecord = namedtuple('SamplesDataRecord',
    field_names=['dset_name', 'data_name', 'data_hash'])

DatasetSchemaRecord = namedtuple('DatasetSchemaRecord',
    field_names=['dset_name', 'schema_hash', 'schema_dtype',
                 'schema_is_var', 'schema_max_shape', 'schema_is_named'])


class DifferBase(object):
    '''Low level class implementing methods common to all record differ objects

    Parameters
    ----------
    ancestor_data : dict
        key/value pairs making up records of the ancestor data
    dev_data : dict
        key/value pairs making up records of the dev data.
    '''

    def __init__(self, ancestor_data: dict, dev_data: dict):
        self.a_data = ancestor_data
        self.d_data = dev_data
        self.a_data_keys = set(self.a_data.keys())
        self.d_data_keys = set(self.d_data.keys())

        self.additions: set = None
        self.removals: set = None
        self.unchanged: set = None
        self.mutations: set = None

    def compute(self, mutation_finder_partial):
        '''perform the computation

        Parameters
        ----------
        mutation_finder_partial : func
            creates a nt descriptor to enable set operations on k/v record pairs.
        '''
        self.additions = self.d_data_keys.difference(self.a_data_keys)
        self.removals = self.a_data_keys.difference(self.d_data_keys)

        a_unchanged_kv, d_unchanged_kv = {}, {}
        potential_unchanged = self.a_data_keys.intersection(self.d_data_keys)
        for k in potential_unchanged:
            a_unchanged_kv[k] = self.a_data[k]
            d_unchanged_kv[k] = self.d_data[k]

        self.mutations = mutation_finder_partial(
            a_unchanged_kv=a_unchanged_kv,
            d_unchanged_kv=d_unchanged_kv)
        self.unchanged = potential_unchanged.difference(self.mutations)

    def kv_diff_out(self):
        '''summary of the changes between ancestor and dev data

        Returns
        -------
        dict
            dict listing all changes via the form: `additions`, `removals`,
            `mutations`, `unchanged`.
        '''
        out = {
            'additions': {k: self.d_data[k] for k in self.additions},
            'removals': {k: self.a_data[k] for k in self.removals},
            'mutations': {k: self.d_data[k] for k in self.mutations},
            'unchanged': {k: self.a_data[k] for k in self.unchanged},
        }
        return out

# -------------------------- Metadata Differ ----------------------------------

class MetadataDiffer(DifferBase):
    '''Specifialized differ class for metadata records.

    Parameters
    ----------
        **kwargs:
            See args of :class:`DifferBase`
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute(self.meta_record_mutation_finder)

    @staticmethod
    def meta_record_mutation_finder(a_unchanged_kv, d_unchanged_kv):

        def meta_nt_func(record_dict: dict) -> set:
            records = set()
            for k, v in record_dict.items():
                records.add(MetaRecord(meta_key=k, meta_hash=v))
            return records

        arecords, drecords = meta_nt_func(a_unchanged_kv), meta_nt_func(d_unchanged_kv)
        mutations = set([m.meta_key for m in arecords.difference(drecords)])
        return mutations


# -------------------- Dataset Schemas Differ ---------------------------------


class DatasetDiffer(DifferBase):
    '''Differ class specifialized for dataset schemas

    Parameters
    ----------
    ancestor_data : dict
        object containing both `data` and `schemas` keys, from
        which only `schemas` will be used in diff
    dev_data : dict
        object containing both `data` and `schemas` keys, from
        which only `schemas` will be used in diff
    '''

    def __init__(self, ancestor_data, dev_data, *args, **kwargs):
        a_schemas = self._isolate_dset_schemas(ancestor_data)
        d_schemas = self._isolate_dset_schemas(dev_data)
        super().__init__(a_schemas, d_schemas, *args, **kwargs)
        self.compute(partial(self.schema_record_mutation_finder,
                             schema_nt_func=self.schema_record_dict_to_nt))

    @staticmethod
    def _isolate_dset_schemas(dataset_specs: dict) -> dict:
        schemas_dict = {}
        for k, v in dataset_specs.items():
            schemas_dict[k] = v['schema']
        return schemas_dict

    @staticmethod
    def schema_record_dict_to_nt(record_dict: dict) -> set:
        records = set()
        for k, v in record_dict.items():
            rec = DatasetSchemaRecord(
                dset_name=k,
                schema_hash=v.schema_hash,
                schema_dtype=v.schema_dtype,
                schema_is_var=v.schema_is_var,
                schema_max_shape=tuple(v.schema_max_shape),
                schema_is_named=v.schema_is_named)
            records.add(rec)
        return records

    @staticmethod
    def schema_record_mutation_finder(schema_nt_func, a_unchanged_kv, d_unchanged_kv):
        arecords, drecords = schema_nt_func(a_unchanged_kv), schema_nt_func(d_unchanged_kv)
        mutations = set([m.dset_name for m in arecords.difference(drecords)])
        return mutations


# ---------------------- Sample Differ ----------------------------------------


class SampleDiffer(DifferBase):
    '''Specialized Differ class for dataset samples.

    Parameters
    ----------
        dset_name: str
            name of the dataset whose samples are being comapared.
        **kwargs:
            See args of :class:`DifferBase`
    '''
    def __init__(self, dset_name: str, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.dset_name = dset_name
        self.compute(self.samples_record_mutation_finder)

    @staticmethod
    def samples_record_mutation_finder(a_unchanged_kv, d_unchanged_kv):

        def samp_nt_func(record_dict: dict) -> set:
            records = set()
            for k, v in record_dict.items():
                rec = SamplesDataRecord(
                    dset_name=k.dset_name, data_name=k.data_name, data_hash=v.data_hash)
                records.add(rec)
            return records

        mutations = set()
        arecords, drecords = samp_nt_func(a_unchanged_kv), samp_nt_func(d_unchanged_kv)
        for m in arecords.difference(drecords):
            rec = parsing.RawDataRecordKey(dset_name=m.dset_name, data_name=m.data_name)
            mutations.add(rec)
        return mutations


# ------------------------- Commit Differ -------------------------------------


class ThreeWayCommitDiffer(object):

    def __init__(self, ancestor_contents: dict, master_contents: dict, dev_contents: dict):

        self.acont = ancestor_contents  # ancestor contents
        self.mcont = master_contents    # master contents
        self.dcont = dev_contents       # dev contents

        self.am_dset_diff: DatasetDiffer = None   # ancestor -> master dset diff
        self.ad_dset_diff: DatasetDiffer = None   # ancestor -> dev dset diff
        self.am_meta_diff: MetadataDiffer = None  # ancestor -> master metadata diff
        self.ad_meta_diff: MetadataDiffer = None  # ancestor -> dev metadata diff
        self.am_samp_diff = {}
        self.ad_samp_diff = {}

        self._run()

    def _run(self):

        self.meta_diff()
        self.dataset_diff()
        self.sample_diff()

# ----------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------

    def meta_diff(self):

        self.am_meta_diff = MetadataDiffer(
            ancestor_data=self.acont['metadata'], dev_data=self.mcont['metadata'])
        self.ad_meta_diff = MetadataDiffer(
            ancestor_data=self.acont['metadata'], dev_data=self.dcont['metadata'])

    def meta_conflicts(self):

        # addition conflicts
        meta_conflicts_t1 = []  # added in master & dev with different values
        addition_keys = self.am_meta_diff.additions.intersection(self.ad_meta_diff.additions)
        for meta_key in addition_keys:
            m_hash = self.am_meta_diff.d_data[meta_key]
            d_hash = self.ad_meta_diff.d_data[meta_key]
            if m_hash != d_hash:
                meta_conflicts_t1.append(meta_key)

        # removal conflicts
        meta_conflicts_t21 = []  # removed in master, mutated in dev
        meta_conflicts_t22 = []  # removed in dev, mutated in master

        am_removal_keys = self.am_meta_diff.removals.intersection(self.ad_meta_diff.mutations)
        meta_conflicts_t21.extend(am_removal_keys)

        ad_removal_keys = self.ad_meta_diff.removals.intersection(self.am_meta_diff.mutations)
        meta_conflicts_t22.extend(ad_removal_keys)

        # mutation conflicts
        meta_conflicts_t311 = []  # mutated in master & dev to different values
        meta_conflicts_t312 = []  # mutated in master, removed in dev
        meta_conflicts_t322 = []  # mutated in dev, removed in master
        for meta_key in self.am_meta_diff.mutations:
            if meta_key in self.ad_meta_diff.mutations:
                m_hash = self.am_meta_diff.d_data[meta_key]
                d_hash = self.ad_meta_diff.d_data[meta_key]
                if m_hash != d_hash:
                    meta_conflicts_t311.append(meta_key)
            elif meta_key in self.ad_meta_diff.removals:
                meta_conflicts_t312.append(meta_key)

        for meta_key in self.ad_meta_diff.mutations:
            if meta_key in self.am_meta_diff.removals:
                meta_conflicts_t322.append(meta_key)

        out = {
            't1': meta_conflicts_t1,
            't21': meta_conflicts_t21,
            't22': meta_conflicts_t22,
            't311': meta_conflicts_t311,
            't312': meta_conflicts_t312,
            't322': meta_conflicts_t322
        }
        conflictFound = False
        for v in out.values():
            if len(v) != 0:
                conflictFound = True
                break
        out['conflict'] = conflictFound
        return out

    def meta_changes(self):
        out = {
            'master': self.am_meta_diff.kv_diff_out(),
            'dev': self.ad_meta_diff.kv_diff_out(),
        }
        return out

    # ----------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------

    def dataset_diff(self):

        self.am_dset_diff = DatasetDiffer(
            ancestor_data=self.acont['datasets'], dev_data=self.mcont['datasets'])
        self.ad_dset_diff = DatasetDiffer(
            ancestor_data=self.acont['datasets'], dev_data=self.dcont['datasets'])

    def dataset_conflicts(self):

        # addition conflicts
        dset_conflicts_t1 = []  # added in master & dev with different values
        for dsetn in self.am_dset_diff.additions:
            if dsetn in self.ad_dset_diff.additions:
                m_srec = self.am_dset_diff.schema_record_dict_to_nt(self.am_dset_diff.d_data[dsetn])
                d_srec = self.ad_dset_diff.schema_record_dict_to_nt(self.ad_dset_diff.d_data[dsetn])
                if m_srec != d_srec:
                    dset_conflicts_t1.append(dsetn)

        # removal conflicts
        dset_conflicts_t21 = []  # removed in master, mutated in dev
        dset_conflicts_t22 = []  # removed in dev, mutated in master
        for dsetn in self.am_dset_diff.removals:
            if dsetn in self.ad_dset_diff.mutations:
                dset_conflicts_t21.append(dsetn)
        for dsetn in self.ad_dset_diff.removals:
            if dsetn in self.am_dset_diff.mutations:
                dset_conflicts_t22.append(dsetn)

        # mutation conflicts
        dset_conflicts_t311 = []  # mutated in master & dev to different values
        dset_conflicts_t312 = []  # mutated in master, removed in dev
        dset_conflicts_t322 = []  # mutated in dev, removed in master
        for dsetn in self.am_dset_diff.mutations:
            if dsetn in self.ad_dset_diff.mutations:
                m_srec = self.am_dset_diff.schema_record_dict_to_nt(self.am_dset_diff.d_data[dsetn])
                d_srec = self.ad_dset_diff.schema_record_dict_to_nt(self.ad_dset_diff.d_data[dsetn])
                if m_srec != d_srec:
                    dset_conflicts_t311.append(dsetn)
            elif dsetn in self.ad_dset_diff.removals:
                dset_conflicts_t312.append(dsetn)

        for dsetn in self.ad_dset_diff.mutations:
            if dsetn in self.am_dset_diff.removals:
                dset_conflicts_t322.append(dsetn)

        out = {
            't1': dset_conflicts_t1,
            't21': dset_conflicts_t21,
            't22': dset_conflicts_t22,
            't311': dset_conflicts_t311,
            't312': dset_conflicts_t312,
            't322': dset_conflicts_t322
        }
        conflictFound = False
        for v in out.values():
            if len(v) != 0:
                conflictFound = True
                break
        out['conflict'] = conflictFound
        return out

    def dataset_changes(self):
        out = {
            'master': self.am_dset_diff.kv_diff_out(),
            'dev': self.ad_dset_diff.kv_diff_out(),
        }
        return out

    # ----------------------------------------------------------------
    # Samples
    # ----------------------------------------------------------------

    def sample_diff(self):

        # ------------ ancestor -> master changes --------------------

        m_dsets = self.am_dset_diff.unchanged.union(
            self.am_dset_diff.additions).union(
                self.am_dset_diff.mutations)
        for dset_name in m_dsets:
            m_dset_data = self.mcont['datasets'][dset_name]['data']
            if dset_name in self.acont['datasets']:
                a_dset_data = self.acont['datasets'][dset_name]['data']
            else:
                a_dset_data = {}
            self.am_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=m_dset_data)

        for dset_name in self.am_dset_diff.removals:
            a_dset_data = self.acont['datasets'][dset_name]['data']
            m_dset_data = {}
            self.am_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=m_dset_data)

        # ------------ ancestor -> dev changes --------------------

        d_dsets = self.ad_dset_diff.unchanged.union(
            self.ad_dset_diff.additions).union(
                self.ad_dset_diff.mutations)
        for dset_name in d_dsets:
            d_dset_data = self.dcont['datasets'][dset_name]['data']
            if dset_name in self.acont['datasets']:
                a_dset_data = self.acont['datasets'][dset_name]['data']
            else:
                a_dset_data = {}
            self.ad_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=d_dset_data)

        for dset_name in self.ad_dset_diff.removals:
            a_dset_data = self.acont['datasets'][dset_name]['data']
            d_dset_data = {}
            self.ad_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=d_dset_data)

    def sample_conflicts(self):

        out = {}
        all_dset_names = set(self.ad_samp_diff.keys()).union(set(self.am_samp_diff.keys()))
        for dsetn in all_dset_names:
            if dsetn in self.am_samp_diff:
                mdiff = self.am_samp_diff[dsetn]
            else:
                mdiff = SampleDiffer(dsetn, {}, {})

            if dsetn in self.ad_samp_diff:
                ddiff = self.ad_samp_diff[dsetn]
            else:
                ddiff = SampleDiffer(dsetn, {}, {})

            # addition conflicts
            samp_conflicts_t1 = []  # added in master & dev with different values
            for samp in mdiff.additions:
                if samp in ddiff.additions:
                    m_rec = mdiff.d_data[samp]
                    d_rec = ddiff.d_data[samp]
                    if m_rec != d_rec:
                        samp_conflicts_t1.append(samp)

            # removal conflicts
            samp_conflicts_t21 = []  # removed in master, mutated in dev
            samp_conflicts_t22 = []  # removed in dev, mutated in master
            for samp in mdiff.removals:
                if samp in ddiff.mutations:
                    samp_conflicts_t21.append(samp)
            for samp in ddiff.removals:
                if samp in mdiff.mutations:
                    samp_conflicts_t22.append(samp)

            # mutation conflicts
            samp_conflicts_t311 = []  # mutated in master & dev to different values
            samp_conflicts_t312 = []  # mutated in master, removed in dev
            samp_conflicts_t322 = []  # mutated in dev, removed in master
            for samp in mdiff.mutations:
                if samp in ddiff.mutations:
                    m_rec = mdiff.d_data[samp]
                    d_rec = ddiff.d_data[samp]
                    if m_rec != d_rec:
                        samp_conflicts_t311.append(samp)
                elif samp in ddiff.removals:
                    samp_conflicts_t312.append(samp)

            for samp in ddiff.mutations:
                if samp in mdiff.removals:
                    samp_conflicts_t322.append(samp)

            out[dsetn] = {
                't1': samp_conflicts_t1,
                't21': samp_conflicts_t21,
                't22': samp_conflicts_t22,
                't311': samp_conflicts_t311,
                't312': samp_conflicts_t312,
                't322': samp_conflicts_t322
            }
            conflictFound = False
            for v in out[dsetn].values():
                if len(v) != 0:
                    conflictFound = True
                    break
            out[dsetn]['conflict'] = conflictFound

        return out

    def sample_changes(self):
        out = {
            'master': {dsetn: self.am_samp_diff[dsetn].kv_diff_out() for dsetn in self.am_samp_diff},
            'dev': {dsetn: self.ad_samp_diff[dsetn].kv_diff_out() for dsetn in self.ad_samp_diff},
        }
        return out

    def all_changes(self, include_master: bool = True, include_dev: bool = True) -> dict:

        meta = self.meta_changes()
        dsets = self.dataset_changes()
        samples = self.sample_changes()

        if not include_master:
            meta.__delitem__('master')
            dsets.__delitem__('master')
            samples.__delitem__('master')
        elif not include_dev:
            meta.__delitem__('dev')
            dsets.__delitem__('dev')
            samples.__delitem__('dev')

        res = {
            'metadata': meta,
            'datasets': dsets,
            'samples': samples,
        }
        return res

    def determine_conflicts(self):
        '''Evaluate and collect all possible conflicts in a repo differ instance

        Parameters
        ----------
        differ : CommitDiffer
            instance initialized with branch commit contents.

        Returns
        -------
        dict
            containing conflict info in `dset`, `meta`, `sample` and `counflict_found`
            boolean field.
        '''
        dset_confs = self.dataset_conflicts()
        meta_confs = self.meta_conflicts()
        sample_confs = self.sample_conflicts()

        conflictFound = False
        for dsetn, confval in sample_confs.items():
            if confval['conflict'] is True:
                conflictFound = True
        if (dset_confs['conflict'] is True) or (meta_confs['conflict'] is True):
            conflictFound = True

        confs = {
            'dset': dset_confs,
            'meta': meta_confs,
            'sample': sample_confs,
            'conflict_found': conflictFound,
        }
        return confs
