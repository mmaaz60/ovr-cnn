import os
from PIL import Image
import jsonlines
import imagesize


class OpenImagesCaptionDataset:
    def __init__(self, root, ann_file, transforms=None, extra_args=None,):
        self._image_root = root
        self._transforms = transforms
        self.metadata = []
        with jsonlines.open(ann_file) as reader:
            for obj in reader:
                self.metadata.append(obj)

    def __getitem__(self, idx):
        fname = self.metadata[idx]['image_id']
        anno = self.metadata[idx]['caption']
        img = Image.open(os.path.join(self._image_root, f"{fname}.jpg")).convert('RGB')
        if self._transforms is not None:
            img, _ = self._transforms(img, None)
        return img, anno, idx

    def get_img_info(self, index):
        img_info = {}
        fname = self.metadata[index]['image_id']
        img_info["width"], img_info["height"] = imagesize.get(os.path.join(self._image_root, f"{fname}.jpg"))
        return img_info

    def __len__(self):
        return len(self.metadata)
