import torch.utils.data as data
from torchvision import transforms


class DeformedData(data.Dataset):
    def __init__(self, data,label):
        '''
        for dataloader
        '''
        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.data = data
        self.label = label
        self.data_index = 0
        self.data_len = len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img_gt = self.label[index]

        return img, img_gt

    def __len__(self):
        return self.data_len


class DeformedTripleData(data.Dataset):
    def __init__(self, data1,data2,data3,label):
        '''
        for dataloader
        '''
        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.label = label
        self.data_index = 0
        self.data_len = len(self.data1)

    def __getitem__(self, index):
        data1 = self.data1[index]
        data2 = self.data2[index]
        data3 = self.data3[index]
        label = self.label[index]

        return data1,data2,data3,label

    def __len__(self):
        return self.data_len