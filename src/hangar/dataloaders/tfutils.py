import random


def yield_data(hangar_datasets, sample_names, shuffle=False):
    if shuffle:
        sample_names = list(sample_names)
        random.shuffle(sample_names)
    for name in sample_names:
        yield tuple([dset[name] for dset in hangar_datasets])
