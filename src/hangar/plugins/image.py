try:
    from PIL import Image
    import numpy as np
except (ModuleNotFoundError, ImportError) as e:
    e.message = ("This plugin you requested has few unmet dependencies. "
                 "Install those before continue")
    raise

from . import ImportExportBase


class ImagePlugin(ImportExportBase):
    def __init__(self):
        super(ImagePlugin, self).__init__()

    @staticmethod
    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def load(self, file):
        # TODO: maybe use accimage in the future
        return np.array(self.pil_loader(file))

    def save(self, file, data):
        data = Image.fromarray(data)
        with open(file, 'wb+') as f:
            data.save(f)
