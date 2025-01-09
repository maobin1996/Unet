import torch
from tqdm import tqdm
from datasets import SegmentationDataset , transform
from torch.utils.data import  DataLoader
from unet import  UNet
import torch.nn as nn
import torch.optim as optim
import argparse
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, save_path='model.pth'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in tqdm(train_loader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # 验证模型
        val_loss = validate_model(model, val_loader, criterion)
        print(f'Validation Loss: {val_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')



def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

    return running_loss / len(val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a U-Net model for image segmentation.')
    
    # 默认路径
    default_image_dir = '/data/common/maobin/VOCdevkit/VOC2012/JPEGImages/'
    default_mask_dir = '/data/common/maobin/VOCdevkit/VOC2012/SegmentationClass/'
    
    parser.add_argument('--image_dir', type=str, default=default_image_dir, help='Path to the directory with images.')
    parser.add_argument('--mask_dir', type=str, default=default_mask_dir, help='Path to the directory with masks.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs for training.')
    parser.add_argument('--model_save_path', type=str, default='model.pth', help='Path to save the trained model.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集和数据加载器
    dataset = SegmentationDataset(args.image_dir, args.mask_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 实例化模型、损失函数和优化器
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, args.model_save_path)

