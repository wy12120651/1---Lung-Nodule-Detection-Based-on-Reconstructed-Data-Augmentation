import os
import sys
import time
import random
import re
import numpy as np
import torch
import argparse
from modules import VectorQuantizedVAE
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F

'''
    - 使用脚本载入训练好的模型，将所有图片进行重构，并存入重构文件夹中
    - 1. 检查/创建 重构图片文件夹
    - 2. 载入训练好的模型, 检查显卡使用, 将模型载入显存中
    - 3. yeild 图片喂给模型, 将输出的tensor进行img转换, 并存储
'''
def align_shape(tensor1, tensor2):
    # 假设 tensor1 是参考目标形状，tensor2 需要对齐
    target_shape = tensor1.shape[2:]  # 获取 tensor1 的第三、四维度（如 H, W）

    # 使用双线性插值调整 tensor2 的尺寸
    aligned_tensor2 = F.interpolate(
        tensor2, 
        size=target_shape,
        mode='bilinear',  # 图像用 'bilinear'，特征图可选 'nearest'/'bicubic'
        align_corners=False  # 是否对齐边角像素，需与模型训练设置一致
    )
    return aligned_tensor2



def save_tensor_as_image(tensor, path, target_size=(256, 256), mean=None, std=None):
    """
    将 [1, C, H, W] 的 Tensor 调整到指定尺寸后保存为图片
    Args:
        tensor: 输入 Tensor (范围需明确，如 [0,1] 或 [-1,1])
        path: 保存路径
        target_size: 目标尺寸 (H, W)
        mean: 归一化均值（若输入已归一化）
        std: 归一化标准差（若输入已归一化）
    """
    tensor = tensor.squeeze(0)
    
    resize_transform = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR)
    tensor = resize_transform(tensor)  # 输出 [C, H_new, W_new]

    if mean is not None and std is not None:
        tensor = tensor * torch.tensor(std).view(-1, 1, 1).to(tensor.device) + torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    
    tensor = (tensor * 255).clamp(0, 255)  
    # 确保值在 0~255
    tensor = tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')  # [H, W, C]

    # Step 5: 保存为图片
    Image.fromarray(tensor).save(path)


def check_name(count):
    
    add_zeros = 4 - len(str(count))
    
    output = '0'*add_zeros + str(count)
        
    return output
    

# check store path of reconstruction imgs

# load model into gpu
parser = argparse.ArgumentParser(description='VQVAE Reproduction')
parser.add_argument('--hidden-size', type=int, default=512, help='size of the latent vectors (default: 256)')
parser.add_argument('--rec_storage', type=str, default='reconstruction_imgs', help='size of the latent vectors (default: 256)')
parser.add_argument('--k', type=int, default=512, help='number of latent vectors (default: 512)')
parser.add_argument('--model_storage', type=str, default='models/vqvae', help='name of the output folder (default: vqvae)')
parser.add_argument('--num_channels', type=int, default=3, help='number of latent vectors (default: 512)')
parser.add_argument('--data-folder', type=str, default='dataset/luna16/all', help='name of the data folder')
parser.add_argument('--device', type=str, default='cuda', help='set the device (cpu or cuda, default: cpu)')

# noise permutation setting
parser.add_argument('--std', type=float, default=0.25, help='noise standard variance value')
parser.add_argument('--mean', type=float, default=0.0, help='noise mean value')

args = parser.parse_args()

args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

model = VectorQuantizedVAE(args.num_channels, args.hidden_size, args.k)
model.load_state_dict(torch.load(args.model_storage + "/best.pt", map_location=args.device))
model.to(device=args.device)

model.eval()

os.makedirs(args.rec_storage, exist_ok=True)



# load the whole image as DataLoader and yelid img-tensor one by one. 
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

dataset = datasets.ImageFolder(root=args.data_folder, transform=transform)

samples = dataset.samples  

original_sizes = []

for path, _ in samples:
    with Image.open(path) as img:
        original_sizes.append(img.size)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,   # 每次返回一个样本
    shuffle=False,  # 是否打乱顺序
    num_workers=0   # 数据加载线程数
)

data_iter = iter(dataloader)
size_iter = iter(original_sizes)

count = 1
try:
    while True:
        inputs, labels = next(data_iter)
        img_size = next(size_iter)
        
        # print(f"输入张量形状: {inputs.shape}, 标签: {labels.item()}")
        
        inputs = inputs.to(args.device)
        z_e_x = model.encoder(inputs)
        z_q_x_st, z_q_x = model.codebook.straight_through(z_e_x)
        
        # 添加噪声，进行编码扰动
        noise = torch.randn_like(z_q_x_st) * args.std + args.mean
        
        z_q_x_st = z_q_x_st + noise.to(args.device)
        
        outputs = model.decoder(z_q_x_st)
        
        outputs = outputs.detach()
        
        # outputs = align_shape(inputs, outputs)
        # outputs = 0.8 * inputs.cpu() + 0.2 * outputs.cpu()
        
        # print(f"输出张量形状: {outputs.shape}, 标签: {labels.item()}")
        
        save_tensor_as_image(tensor=outputs, path=args.rec_storage + '/'+ check_name(count=count) + '.png', target_size=img_size)
        
        count += 1

except StopIteration:
    print("数据已遍历完毕")







if __name__ == '__main__':
    
    ...