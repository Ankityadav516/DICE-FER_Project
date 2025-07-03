import torch
import torch.nn.functional as F

def l1_loss(e1, e2):
    """
    L1 reconstruction loss between two feature representations.
    """
    return F.l1_loss(e1, e2)

def adv_loss(real_score, fake_score):
    """
    Adversarial loss using binary cross-entropy.
    real_score: output of discriminator for real (matched) pairs
    fake_score: output of discriminator for mismatched pairs
    """
    eps = 1e-8  # for numerical stability
    return -torch.mean(torch.log(real_score + eps) + torch.log(1 - fake_score + eps))
