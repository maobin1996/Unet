import torch
import numpy as np
import argparse
from datasets import SegmentationDataset, transform
from torch.utils.data import DataLoader
from unet import UNet
import os  
from PIL import Image  
def test_model(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.float().to(device)
            print("image size is ", images.size())
            outputs = model(images)
            print("shape :", outputs.shape) 
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions, axis=0)

def custom_collate_fn(batch):  
    # 过滤掉 `None` 元素，只留下有效图像  
    images = [item[0] for item in batch if item[0] is not None]  # 只取第一个元素  
    return torch.stack(images)  # 将图像堆叠成一个批次

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a U-Net model for image segmentation.')
    
    # 默认路径
    default_image_dir = 'imgs'
    
    parser.add_argument('--image_dir', type=str, default=default_image_dir, help='Path to the directory with images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集和数据加载器
    dataset = SegmentationDataset(args.image_dir, transform=transform, is_test=True)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,collate_fn=custom_collate_fn)

    # 实例化模型并加载训练好的权重
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path))
    
    # 测试模型
    predictions = test_model(model, test_loader)
    print("predictions shape:", predictions.shape)
    #保存图片
    threshold = 0.1  # 设置阈值  
    for i, pred in enumerate(predictions):  
        pred_binary = (pred.squeeze() > threshold).astype(np.uint8)  # 转换为二值图像  
        img = Image.fromarray(pred_binary * 255)  # 将二值数组转为图像、乘以255以使图像像素范围[0, 255]  
        output_path = os.path.join("imgs", f'prediction_{i}.png')  # 指定保存路径  
        img.save(output_path)  # 保存图像  

    print("Testing completed. Predictions shape:", predictions.shape)
