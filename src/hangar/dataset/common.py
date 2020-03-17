# TODO: document that Dataset can only load ndarray


class Dataset:
    def __init__(self, columns, keys):
        if not isinstance(columns, (list, tuple, set)):
            columns = (columns,)
        if len(columns) == 0:
            raise ValueError('len(columns) cannot == 0')
        all_keys = []
        all_remote_keys = []
        for col in columns:
            if col.iswriteable is True:
                raise TypeError(f'Cannot load columns opened in `write-enabled` checkout.')
            all_keys.append(set(col.keys()))
            all_remote_keys.append(set(col.remote_reference_keys))
        common_keys = set.intersection(*all_keys)
        remote_keys = set.union(*all_remote_keys)
        common_local_keys = common_keys.difference(remote_keys)
        if keys:
            if not isinstance(keys, (list, tuple, set)):
                raise TypeError('keys must be a list/tuple/set of hangar sample keys')
            unique = set(keys)
            notCommon = unique.difference(common_keys)
            notLocal = unique.difference(common_local_keys)

            # TODO: These error message could eat up the whole terminal space if the size of
            #   non common and non local keys are huge
            if len(notCommon) > 0:
                raise KeyError(f'Keys: {notCommon} do not exist in all columns.')
            if len(notLocal) > 0:
                raise FileNotFoundError(
                    f'Keys: {notLocal} are remote data samples not downloaded locally.')
        else:
            keys = common_local_keys
        if len(keys) == 0:
            raise ValueError('No Samples available common to all '
                             'columns and available locally.')

        # TODO: May be we need light weight data accessors instead of columns
        self.columns = columns
        self.keys = list(keys)

    def __getitem__(self, item):
        return tuple([col[item] for col in self.columns])
