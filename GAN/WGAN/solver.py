import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import seaborn as sns
from visdom import Visdom
import numpy as np
import os
from tqdm import tqdm
from models import Discriminator, Generator

def to_var(x, eval=False):
    """Tensor => Variable"""
    if torch.cuda.is_available():
        x = x.cuda()
    if eval:
        x = Variable(x, volatile=True)
    else:
        x = Variable(x)
    return x

def to_tensor(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

def denorm(x):
    """(-1, 1) => (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)

def initialize(model):
    """Apply Xavier initilization on weight parameters"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            init.xavier_normal(module.weight.data)

def sample(*size):
    """Sampling noise from gaussian distribution"""
    z = to_var(torch.randn(*size))
    return z


class Solver(object):
    def __init__(self, config, data_loader, is_train=False):
        self.config = config
        self.data_loader = data_loader
        self.n_batches_in_epoch = len(self.data_loader)
        self.total_data_size = len(self.data_loader.dataset)
        self.is_train = is_train


    def build(self):
        """Build Networks, losses and optimizers(training only)
        if torch.cuda.is_available() is True => Use GPUs
        """
        self.D = Discriminator(self.config)
        self.G = Generator(self.config)

        if self.config.cuda:
            self.D.cuda()
            self.G.cuda()

        # Build Optimizer (Training Only)
        if self.is_train:
            self.d_optimizer = self.config.optimizer(
                self.D.parameters(),
                lr=self.config.d_learning_rate,
                )
                # betas=(self.config.beta1, self.config.beta2))
            self.g_optimizer = self.config.optimizer(
                self.G.parameters(),
                lr=self.config.g_learning_rate,
                )
                # betas=(self.config.beta1, self.config.beta2))

    def d_weight_clamp(self):
        """Clamp weights of Discriminator (K-Lipschitz condition)"""
        for p in self.D.parameters():
            p.data.clamp_(-self.config.w_clamp, self.config.w_clamp)

    def d_loss(self, x, z):
        """Loss for Discriminator
        Earth-Mover distance =   ( mean_i ( fw(x(i)) ) − mean_i ( fw(gθ(z(i))) ) (no logarithm)
        Loss = - EM distance = - ( mean_i ( fw(x(i)) ) − mean_i ( fw(gθ(z(i))) )

        Return:
            d_loss: [1]
            D(x): [batch_size, 1]
            D(G(z)): [batch_size, 1]
        """
        # Comput loss with real images
        D_x = self.D(x) # [batch_size, 1]; Discriminator's classification on real images

        # Compute loss with fake images
        G_z = self.G(z) # generate image from noise
        D_G_z = self.D(G_z) # Discriminator's classification on fake images

        d_loss = -(torch.mean(D_x) - torch.mean(D_G_z))

        return d_loss, D_x, D_G_z

    def g_loss(self, z):
        """Loss for Generator
        - mean_i ( fw(gθ(z(i)) )

        Return:
            g_loss: [1]
        """

        # Compute loss with fake images
        G_z = self.G(z) # generate image from noise
        D_G_z = self.D(G_z) # Discriminator's classification on fake images
        g_loss = -torch.mean(D_G_z)

        return g_loss

    def save_images(self, epoch, x, z):
        """Save intermediate results for debugging"""

        batch_size = x.size(0)

        # Save real images (first epoch only)
        if epoch == 1:
            self.fixed_images = x # for debugging
            self.fixed_noise = z # for debugging

            real_images = self.fixed_images.view(batch_size, 1, 28, 28).data
            real_image_path = os.path.join(self.config.image_log_path, 'real_images.png')
            print('Save real images at %s' % (real_image_path))
            save_image(real_images, real_image_path)

        # Save fake images
        if epoch % self.config.save_interval == 0:
            # fill zeros (ex. 1 => 001)
            epoch = str(epoch).zfill(3)

            fake_images = self.G(self.fixed_noise)
            fake_images = fake_images.view(batch_size, 1, 28, 28).data
            # fake_images = denorm(fake_images.data) # (-1, 1) => (0, 1)
            fake_image_path = os.path.join(self.config.image_log_path, f'fake_images-{epoch}.png')
            print(f'Save fake images at {fake_image_path}')
            save_image(fake_images, fake_image_path)

    def save_model(self, epoch):
        """Save parameters at checkpoint"""

        d_ckpt_path = os.path.join(self.config.save_dir, f'D_epoch-{epoch}.pkl')
        print(f'Save parameters at {d_ckpt_path}')
        torch.save(self.D.state_dict(), d_ckpt_path)

        g_ckpt_path = os.path.join(self.config.save_dir, f'G_epoch-{epoch}.pkl')
        print(f'Save parameters at {g_ckpt_path}\n')
        torch.save(self.G.state_dict(), g_ckpt_path)


    def train(self):
        """Training Discriminator and Generator

        1. Initialize Discriminator and Generator.
        2. Set Discriminator and Generator to training mode.
        3. for epoch in epochs:
            for batch in batches:
                Get batch from data_loader.
                Train Discriminator.
                Train Generator.
                Log training details every predefined batches.
            Save original images only at the first epoch.
            Save generated images.
            Pickle parameters."""
        print('Start Training!\n')

        # Initialize parameters
        initialize(self.D)
        initialize(self.G)

        # Set to training mode
        self.D.train()
        self.G.train()

        self.D_loss_history = []
        self.G_loss_history = []

        for epoch in tqdm(range(self.config.epochs)):
            epoch += 1
            epoch_d_loss = 0
            epoch_g_loss = 0
            for batch_idx, (images, _) in enumerate(self.data_loader):
                # images: [batch_size, 1, 28, 28]
                batch_idx += 1
                batch_size = images.size(0)

                # [batch_size, 1, 28, 28] => [batch_size, 784]
                x = to_var(images.view(batch_size, -1))

                ###########################
                #   Train Discriminator   #
                ###########################

                for _ in range(self.config.d_n_iter):
                    # Sampling noise from gaussian distribution
                    z = sample(batch_size, self.config.z_size)

                    # Caculate Loss
                    d_loss, D_x, D_G_z = self.d_loss(x, z)

                    # Backprop & Update
                    self.D.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Weight Clipping (K-Lipschitz condition)
                    self.d_weight_clamp()

                #######################
                #   Train Generator   #
                #######################

                # Sampling noise from gaussian distribution
                z = sample(batch_size, self.config.z_size)

                # Calculate Loss
                g_loss = self.g_loss(z)

                # Backprop & Update
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Variable => Tensor => float
                d_loss = to_tensor(d_loss)[0]
                g_loss = to_tensor(g_loss)[0]
                D_x = to_tensor(D_x).mean() # Discriminator's classification among batch
                D_G_z = to_tensor(D_G_z).mean() # Discriminator's classification among batch

                # Epoch loss
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss

                # Print Training details every `log_interval`
                if batch_idx > 1 and batch_idx % self.config.log_interval == 0:
                    progress_percentage = 100. * batch_idx / self.n_batches_in_epoch
                    average_batch_d_loss = d_loss / batch_size
                    average_batch_g_loss = g_loss / batch_size

                    log_string = f'Epoch [{epoch}/{self.config.epochs}]  Step [{batch_idx}/{self.n_batches_in_epoch}] '
                    log_string += f'({progress_percentage:.0f}%) | '
                    log_string += f'D Loss: {average_batch_d_loss:.3f} | '
                    log_string += f'G Loss: {average_batch_g_loss:.3f} | '
                    log_string += f'D(x): {D_x:.3f} | '
                    log_string += f'D(G(z)): {D_G_z:.3f}'

                    print(log_string)

            # Epoch summary
            average_epoch_d_loss = epoch_d_loss / self.total_data_size
            average_epoch_g_loss = epoch_g_loss / self.total_data_size
            print(f'\nEpoch {epoch} | D Loss: {average_epoch_d_loss:.3f}')
            print(f'Epoch {epoch} | G Loss: {average_epoch_g_loss:.3f}\n')
            self.D_loss_history.append(average_epoch_d_loss)
            self.G_loss_history.append(average_epoch_g_loss)

            self.visdom_plot(epoch, average_epoch_d_loss, average_epoch_g_loss)

            # Save real / fake images at every config.save_interval epochs
            self.save_images(epoch, x, z)

            # Save parameters at checkpoint
            self.save_model(epoch)

    def eval(self):

        max_epoch = str(self.config.epochs).zfill(3)

        print('Start Evaluation!\n')
        # Load Parameters from checkpoint
        d_ckpt_path = os.path.join(self.config.save_dir, f'D_epoch-{max_epoch}.pkl')
        print(f'Load parameters from {d_ckpt_path}')
        self.D.load_state_dict(torch.load(d_ckpt_path))

        g_ckpt_path = os.path.join(self.config.save_dir, f'G_epoch-{max_epoch}.pkl')
        print(f'Load parameters from {g_ckpt_path}')
        self.G.load_state_dict(torch.load(g_ckpt_path))

        # Set to evaluation mode
        self.D.eval()
        self.G.eval()

        d_loss = 0
        g_loss = 0

        for batch_idx, (images, _) in enumerate(self.data_loader):
            batch_size = images.size(0)

            x = to_var(images.view(batch_size, -1), eval=True)
            z = sample(batch_size, self.config.z_size)

            d_loss, _, _ = self.d_loss(x, z)
            g_loss += self.g_loss(z)


        d_loss /= self.total_data_size
        g_loss /= self.total_data_size

        print(f'D Loss: {d_loss.data[0]:.3f}')
        print(f'G Loss: {g_loss.data[0]:.3f}\n')

    def plot(self):
        """plot learning curve with matplotlib + seaborn
        X: Epoch
        Y: D_loss, G_loss"""

        fig_path = os.path.join(self.config.image_log_path, 'learning_curve.png')
        fig = plt.figure(figsize=(20,8))
        x = range(1, len(self.D_loss_history)+1)

        with sns.axes_style('whitegrid'):
            plt.plot(x, self.D_loss_history, label='Discriminator')
            plt.plot(x, self.G_loss_history, label='Generator')
            plt.title('Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            plt.xlim(1,)
            fig.savefig(fig_path)

    def visdom_plot(self, epoch, d_loss, g_loss):
        """plot learning curve interactively with visdom
        X: Epoch
        Y: D_loss, G_loss"""

        if epoch == 1:
            self.vis = Visdom(env='WGAN')

            self.win = self.vis.line(
                X=np.array([epoch]),
                Y=np.array([[d_loss, g_loss]]),
                opts=dict(
                    title='Learning Curve',
                    xlabel='Epoch',
                    ylabel='Loss',
                    legend=['Discriminator', 'Generator'],
                ))
        else:
            self.vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([d_loss]),
                win=self.win,
                name='Discriminator',
            )
            self.vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([g_loss]),
                win=self.win,
                name='Generator',
            )
