import torch.nn.functional as F

def l1_loss(e1, e2):
    return F.l1_loss(e1, e2)

def adv_loss(real_score, fake_score):
    return -torch.mean(torch.log(real_score + 1e-8) + torch.log(1 - fake_score + 1e-8))

