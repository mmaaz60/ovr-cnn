import os
import json
from PIL import Image


class OpenImagesCaptionDataset:
    def __init__(self, root, ann_file, transforms=None, extra_args=None,):
        self._image_root = root
        self._transforms = transforms
        with open(ann_file, 'r') as fin:
            self.metadata = json.load(fin)

    def __getitem__(self, idx):
        fname = self.metadata[idx]['image_id']
        anno = self.metadata[idx]['caption']
        img = Image.open(os.path.join(self._image_root, fname)).convert('RGB')
        if self._transforms is not None:
            img, _ = self._transforms(img, None)
        return img, anno, idx

    def get_img_info(self, index):
        return self.metadata[index]

    def __len__(self):
        return len(self.metadata)
