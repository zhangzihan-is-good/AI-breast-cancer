import torch
import torch.nn.functional as F


def gaussian_window(size, sigma):
    """ Create a Gaussian window """
    x = torch.arange(size, dtype=torch.float32) - size // 2
    gauss = torch.exp(-(x**2) / float(2 * sigma**2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """ Create a 2D window for SSIM calculation """
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """ Calculate SSIM between two images """
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
    
# The loss of VAE is a combination of MSE loss and SSIM loss
def ae_loss(recon_x, x):
    # MSE loss
    MSE_loss = F.mse_loss(recon_x, x, reduction='mean')
    # SSIM loss
    ssim_out = 1 - ssim(recon_x, x)
    return MSE_loss + ssim_out
