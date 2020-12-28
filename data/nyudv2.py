from torch.utils import data
from torchvision import transforms

from data.transform.rgbd_transform import *

def make_dataset_fromlst(listfilename):
    """
    NYUlist format: image_path label_path depth_path HHA_path
    Args:
        listfilename: file path of list
    """
    images = []
    segs = []
    depths = []
    HHAs = []

    with open(listfilename) as f:
        content = f.readlines()
        for x in content:
            imgname, segname, depthname, HHAname = x.strip().split(' ')
            images += [imgname]
            segs += [segname]
            depths += [depthname]
            HHAs += [HHAname]

        return {'images':images, 'segs':segs, 'HHAs':HHAs, 'depths':depths}


class NYUDataset_val_full(data.Dataset):
    """
    NYUDataset for evaluation with full size
    Init Args:
        list_path: file path of NYUlist
    """
    def __init__(self, list_path):
        self.list_path = list_path
        self.paths_dict = make_dataset_fromlst(self.list_path)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):
        # self.paths['images'][index]
        img = Image.open(self.paths_dict['images'][index])  # .astype(np.uint8)
        depth = Image.open(self.paths_dict['depths'][index])
        HHA = Image.open(self.paths_dict['HHAs'][index])
        seg = Image.open(self.paths_dict['segs'][index])

        sample = {'image':img,
                  'depth':depth,
                  'seg': seg,
                  'HHA': HHA}

        sample = self.transform_val(sample)
        sample = self.totensor(sample)

        return sample

    def __len__(self):
        return self.len

    def name(self):
        return 'NYUDataset_val_full'

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            Normalize_PIL2numpy_depth2xyz()])
        return composed_transforms(sample)

    def totensor(self, sample):
        composed_transforms = transforms.Compose([
            ToTensor()])
        return composed_transforms(sample)