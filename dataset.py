import math
import os
from loguru import logger
from threading import Thread
import torch.utils.data
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms


class ClassficationDataset(Dataset):
    def __init__(self, path, transform=None, cache_img=False, target=None):
        self.path = path
        self.cache_img = cache_img
        self.transform = transform
        self.img_paths_label = []
        self.img_labels = None

        class_dirs = os.listdir(self.path)
        class_dirs.sort()
        if target is not None and set(class_dirs) == set(target):
            class_dirs = target
        else:
            logger.info('Directory category and target are not the same, use default category' + str(class_dirs))
        class_dirs = [os.path.join(self.path, class_dir) for class_dir in class_dirs]

        for index, class_dir in enumerate(class_dirs):
            img_names = os.listdir(class_dir)
            img_names.sort()

            for name in img_names:
                img_path = os.path.join(class_dir, name)
                self.img_paths_label.append([img_path, index])
        logger.info('total images:' + str(len(self.img_paths_label)))

        if cache_img:
            logger.info('caching images ...')
            group = len(self.img_paths_label) / 4
            group = math.ceil(group)
            img_paths_label_groups = [self.img_paths_label[i:i + group] for i in range(0, len(self.img_paths_label), group)]

            threads = [ReadImg(readImg, args=(img_paths_label_groups[i], i)) for i in range(4)]
            [thread.start() for thread in threads]
            [thread.join() for thread in threads]

            img_labels = []
            [img_labels.extend(thread.get_result()) for thread in threads]
            self.img_labels = img_labels

    def __len__(self):
        return len(self.img_paths_label)

    def __getitem__(self, item):
        if self.cache_img:
            img_label = self.img_labels[item]
            img = img_label[0]
            img = self.transform(img) if self.transform is not None else img
            label = img_label[1]
        else:
            img_path_label = self.img_paths_label[item]
            img_path = img_path_label[0]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img) if self.transform is not None else img
            label = img_path_label[1]

        return img, label


class ReadImg(Thread):
    def __init__(self, func, args):
        super(ReadImg, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def readImg(img_infos, index):
    img_labels = []
    for img_info in img_infos:
        img_path = img_info[0]
        label = img_info[1]
        img_labels.append([Image.open(img_path).convert('RGB'), label])
    return img_labels


if __name__ == "__main__":
    transform = [transforms.Resize([224, 224]),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    dataset = ClassficationDataset('data/cifar/test', transform=transforms.Compose(transform), cache_img=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=8, shuffle=False)