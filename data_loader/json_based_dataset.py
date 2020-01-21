import json
import sys

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension, default_loader
import pandas as pd
import glob
import os
import re


# def extract_label(img_path):
#     """
#     Extract label from "image path"
#
#     arguments:
#     img_path (string): ex)"images/Egyptian_Mau_63.jpg"
#
#     return:
#     label (string): ex)Egyption_Mau
#     """
#     label = re.search(r'images\\(.*)(_(.+).jpg)', img_path).group(1)
#     return label
#
#
# def make_json_dataset_from_folder(raw_data_dir, output_path):
#     image_paths = glob.glob("{}/*.jpg".format(raw_data_dir))
#     image_df = pd.DataFrame(image_paths)
#     image_df.columns = ['img_path']
#     image_df['label'] = image_df.img_path.map(extract_label)
#     output = os.path.join(output_path, 'json_data_set.json')
#     image_df.to_json(output, orient='values')
#
#
# #make_json_dataset_from_folder("images")


def find_classes(root_dir):
    """
    Finds the class folders in a dataset.

    Args:
        root_dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_json_from_labeled_folder(root_dir, output_path, is_valid_file=None):
    root_dir = os.path.expanduser(root_dir)
    extensions = IMG_EXTENSIONS

    classes, class_to_idx = find_classes(root_dir)
    path_dict = {cls: [] for cls in classes}
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    path_dict[target].append(path)

    output = os.path.join(output_path)

    with open(output, 'w') as json_file:
        json.dump(path_dict, json_file)


def make_dataset_from_json(root_dir, json_file_path, extensions=None, is_valid_file=None):
    images = []
    json_data = None
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    classes = [key for key in json_data.keys()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for key in json_data:
        for path in json_data[key]:
            d = os.path.join(root_dir, path)
            if os.path.exists(d):
                if is_valid_file(d):
                    item = (d, class_to_idx[key])
                    images.append(item)

    return images, classes, class_to_idx


class JsonBasedDataset(VisionDataset):
    def __init__(self, root, json_file_path, loader=default_loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(JsonBasedDataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None

        self.transform = transform
        self.target_transform = target_transform

        samples, classes, class_to_idx = make_dataset_from_json(root, json_file_path, self.extensions)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
