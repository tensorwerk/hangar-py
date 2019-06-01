class DataLoader(object):
    '''
    Base class for dataloaders. All the necessary APIs required for
    a dataloader class must be created here. But the high level APIs
    such as batching will be available in child classes. This class
    must be inherited to create custom loaders for different integrations
    as well such as PyTorch loader.
    '''

    def __init__(self, dataset):
        pass

    def __len__(self):
        pass

    def __getitem__(self, key):
        pass


class TorchDataLoader(DataLoader):
    pass


class TensorflowDataLoader(DataLoader):
    pass
