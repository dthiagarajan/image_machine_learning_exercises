from pt_methods import *
import torch.utils.data as data


class MedicalImages(data.Dataset):

    def __init__(self, data, labels,
                 transform=None, target_transform=None):

        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # Image from array needs a hwc format, although 
        # pytorch will convert it back to chw...
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
