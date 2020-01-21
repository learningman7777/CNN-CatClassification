from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import JsonBasedDataset

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.datasets as vdatasets
import os

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CatClassificationDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
        data_transforms = transforms.Compose([
                transforms.Resize((197, 197)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.data_dir = data_dir
        self.dataset = JsonBasedDataset("", 'data.json', transform=data_transforms)
        self.label_list = self.dataset.classes
        self.int_to_label_dict = {v: k for k, v in self.dataset.class_to_idx.items()}

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# class CatClassificationDataLoader(DataLoader):
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, collate_fn=default_collate):
#         self.validation_split = validation_split
#         self.shuffle = shuffle
#
#         self.batch_idx = 0
#
#
#         data_transforms={
#             'TRAIN': transforms.Compose([
#                 transforms.Resize((224, 224)),  # 1. 사이즈를 224, 224로 통일.
#                 transforms.RandomHorizontalFlip(),  # 좌우반전으로 데이터셋 2배 뻥튀기
#                 transforms.ToTensor(),  # 2. PIL이미지를 숫자 텐서로 변환.
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 3. 노멀라이즈
#             ]),
#             'VAL': transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#         }
#
#         # 이미지 데이터셋의 형태로 트레이닝과 밸리데이션 데이터셋을 준비합니다.
#         # 이 ImageFolder클래스에 폴더를 집어 넣으면, raw이미지를 읽어서 데이터셋을 만들어 주는데,
#         # 이 때, 폴더명이 classname(Supervised Learning에서 Label)이 됩니다.
#         image_datasets = {x: vdatasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                           for x in ['TRAIN', 'VAL']}
#
#         self.label_list = image_datasets['TRAIN'].classes
#         self.int_to_label_dict = {v: k for k, v in image_datasets['TRAIN'].class_to_idx.items()}
#
#         self.n_samples = len(image_datasets['TRAIN'])
#
#         self.train_sampler = image_datasets['TRAIN']
#         self.valid_sampler = image_datasets['VAL']
#
#         self.init_kwargs = {
#             'batch_size': batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': collate_fn,
#             'num_workers': num_workers
#         }
#         super().__init__(self.train_sampler, **self.init_kwargs)
#
#     def split_validation(self):
#         if self.valid_sampler is None:
#             return None
#         else:
#             return DataLoader(self.valid_sampler, **self.init_kwargs)
