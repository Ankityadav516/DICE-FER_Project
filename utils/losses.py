import torch
import torch.nn.functional as F

def l1_loss(e1, e2):
    """
    L1 reconstruction loss between two feature representations.
    """
    return F.l1_loss(e1, e2)

def adv_loss(real_score, fake_score):
    """
    Adversarial loss using LSGAN (least-squares GAN):
    Encourages real_score → 1 and fake_score → 0.
    More stable than BCE-based GAN loss.
    """
    real_loss = F.mse_loss(real_score, torch.ones_like(real_score))
    fake_loss = F.mse_loss(fake_score, torch.zeros_like(fake_score))
    return real_loss + fake_loss
