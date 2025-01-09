import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

import os  
from PIL import Image  
from torch.utils.data import Dataset  

class SegmentationDataset(Dataset):  
    def __init__(self, image_dir, mask_dir=None, transform=None, is_test=False):  
        self.image_dir = image_dir  
        self.mask_dir = mask_dir  
        self.transform = transform  
        self.images = os.listdir(image_dir)  # 读取图像文件名  
        self.is_test = is_test  # 新增一个参数以标识是否是测试模式  

    def __len__(self):  
        return len(self.images)  

    def __getitem__(self, idx):  
        # 加载图像  
        img_path = os.path.join(self.image_dir, self.images[idx])  
        image = Image.open(img_path).convert("RGB")  

        if self.is_test:  # 如果是测试模式，不加载掩码  
            if self.transform:  
                image = self.transform(image)  
            return image, None  # 返回图像和 None 作为掩码  
        else:  
            # 加载掩码  
            mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))  # 假设掩码文件名与图像相关  
            mask = Image.open(mask_path).convert("L")  # 通道为1的灰度图  

            if self.transform:  
                image = self.transform(image)  
                mask = self.transform(mask)  

            return image, mask

# class SegmentationDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = os.listdir(image_dir)  # 读取图像文件名

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         # 加载图像和掩码
#         img_path  = os.path.join(self.image_dir, self.images[idx])
#         mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))  # 假设掩码文件名与图像相关
        
#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")  # 通道为1的灰度图

#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         return image, mask

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
])

# # 创建数据集和数据加载器
# image_dir = 'C:\\Users\\mao\\Desktop\\deeplearning\\VOC2012\\JPEGImages'  # 图像文件夹路径
# mask_dir = 'C:\\Users\\mao\\Desktop\\deeplearning\\VOC2012\\SegmentationClass'  # 掩码文件夹路径

# dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # 遍历数据加载器
# for images, masks in dataloader:
#     print(images.shape, masks.shape)  # 输出图像和掩码的形状



