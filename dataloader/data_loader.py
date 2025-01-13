from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
from util import task
import random
import math
import os

class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        #self.img_paths, self.img_size = make_dataset(opt.img_file)

        path = self.opt.img_file

        self.mask_path = os.path.join(path, "masks")
        self.normal_path = os.path.join(path, "normal")

        self._all_fnames = os.listdir(self.normal_path)


        self.mask_paths = [os.path.join(self.mask_path, nodule_path) for nodule_path in os.listdir(self.mask_path)]
        self.normal_paths = [os.path.join(self.normal_path, nodule_path) for nodule_path in os.listdir(self.normal_path)]

        self.transform = get_transform(opt)

        self.img_size = len(self.normal_paths)

        #print(self.opt.fineSize)

        self.mask_transform = transforms.Compose([
          transforms.Resize(self.opt.fineSize),
          transforms.ToTensor()
        ])



    def __getitem__(self, index):
        img_path = self.normal_paths[index]
        img = self.transform(Image.open(img_path).convert("L"))
        mask = self.mask_transform(Image.open(random.choice(self.mask_paths)).convert("L"))

        return {'img': img, 'img_path': img_path, 'mask': mask}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    """
    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path

    def load_mask(self, img, index):
        #Load different mask types for training and testing
        mask_type_index = random.randint(0, len(self.opt.mask_type) - 1)
        mask_type = self.opt.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return task.center_mask(img)

        # random regular mask
        if mask_type == 1:
            return task.random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return task.random_irregular_mask(img)

        if mask_type == 4:
            return task.random_freefrom_mask(img)

        # external mask from "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV18)"
        if mask_type == 3:
            mask_index = index
            mask_pil = Image.open(self.mask_paths[mask_index]).convert('L')
            size = mask_pil.size[0]
            if size > mask_pil.size[1]:
                size = mask_pil.size[1]
            
            

            # Directly dotted with image in model forward so that means 0 is part of image being removed and that matches NoduleGen dataset so no need for == 0
            mask = (mask_transform(mask_pil)).float()
            mask_pil.close()
            return mask
    
    """


def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=not opt.no_shuffle, num_workers=int(opt.nThreads))

    return dataset


def get_transform(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    
    transform_list.append(transforms.Resize(fsize))
    transform_list.append(transforms.Grayscale())

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)
