import torch, sys 
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms, datasets
import argparse

transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def stack_tensor(path:str, transform=transform):
    
    real_dataset = datasets.ImageFolder(root=path, transform=transform)

    all_images = [image for image, _ in real_dataset] 
    stacked_tensor = torch.stack(all_images, dim=0)
    return stacked_tensor

def psnr(original, reconstructed, max_pixel=1.0):
    ''' 越高越好
    '''

    mse = torch.mean((original - reconstructed) ** 2)
    return 10 * torch.log10(max_pixel**2 / mse)


def get_inception_features(images, model, device):
    model.eval()
    with torch.no_grad():
        # 输入图像需调整为299x299且归一化为[0,1]
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear')
        features = model(images)  # 获取Inception-v3的最后一层池化前特征

    return features.cpu().numpy()

def calculate_fid(real_features, fake_features):
    ''' 越小越好
    '''
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    diff = mu_real - mu_fake
    cov_mean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * cov_mean)
    return fid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metrics')
    parser.add_argument('--hidden-size', type=int, default=512, help='size of the latent vectors (default: 256)')
    parser.add_argument('--rec_storage', type=str, default='reconstruction_imgs/', help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512, help='number of latent vectors (default: 512)')
    parser.add_argument('--model_storage', type=str, default='models/vqvae', help='name of the output folder (default: vqvae)')
    parser.add_argument('--num_channels', type=int, default=3, help='number of latent vectors (default: 512)')
    parser.add_argument('--data-folder', type=str, default='dataset/luna16/all', help='name of the data folder')
    parser.add_argument('--device', type=str, default='cuda', help='set the device (cpu or cuda, default: cpu)')

    # noise permutation setting
    parser.add_argument('--std', type=float, default=0.25, help='noise standard variance value')
    parser.add_argument('--mean', type=float, default=0.0, help='noise mean value')

    args = parser.parse_args()
    # 使用示例
    
    original = stack_tensor(path=args.data_folder)
    
    reconstructed = stack_tensor(path=args.rec_storage)
    
    # PSNR用例
    print("PSNR:", psnr(original, reconstructed))

    # SSIM 越大越好
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    score = ssim(original, reconstructed)
    print("SSIM:", score)

    sys.exit(1)

    # FID用例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
    inception_model.fc = torch.nn.Identity()  # 禁用全连接层
    real_features = get_inception_features(original, inception_model, device)
    fake_features = get_inception_features(reconstructed, inception_model, device)
    print("FID:", calculate_fid(real_features, fake_features))
