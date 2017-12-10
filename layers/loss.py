import torch
import torch.nn as nn
from torch.autograd import Variable


class VAELoss(nn.Module):
    def __init__(self, recon_loss='l2'):
        """VAE Loss = Reconsturction Loss + KL Divergence"""
        super(VAELoss, self).__init__()

        if recon_loss == 'bce':
            self.recon_loss = nn.BCELoss(size_average=False)
        elif recon_loss == 'l2':
            self.recon_loss = nn.MSELoss(size_average=False)
        elif recon_loss == 'l1':
            self.recon_loss = nn.L1Loss(size_average=False)

    def kl_div(self, mu, log_variance):
        """KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)"""
        return -0.5 * torch.sum(1 + log_variance - mu**2 - log_variance.exp())

    def __call__(self, x, recon_x, mu, log_variance):
        recon_loss = self.recon_loss(recon_x, x)
        kl_div = self.kl_div(mu, log_variance)
        return recon_loss, kl_div


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
