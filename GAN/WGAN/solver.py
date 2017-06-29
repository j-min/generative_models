import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import seaborn as sns
from visdom import Visdom
import imageio
from scipy import ndimage
import numpy as np
import os
from tqdm import tqdm
from models import MLP_D, MLP_G, DCGAN_D, DCGAN_G

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
            init.xavier_normal(module.weight.data, gain=0.5)

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
        if self.config.DCGAN:
            self.D = DCGAN_D(self.config)
            self.G = DCGAN_G(self.config)
        else:
            self.D = MLP_D(self.config)
            self.G = MLP_G(self.config)

        if self.config.cuda:
            self.D.cuda()
            self.G.cuda()

        # Build Optimizer (Training Only)
        if self.is_train:
            self.D_optimizer = self.config.optimizer(
                self.D.parameters(),
                lr=self.config.D_learning_rate,
                )
                # betas=(self.config.beta1, self.config.beta2))
            self.G_optimizer = self.config.optimizer(
                self.G.parameters(),
                lr=self.config.G_learning_rate,
                )
                # betas=(self.config.beta1, self.config.beta2))

    def D_weight_clamp(self):
        """Clamp weights of Discriminator (K-Lipschitz condition)"""
        for p in self.D.parameters():
            p.data.clamp_(-self.config.D_weight_clamp, self.config.D_weight_clamp)

    def D_loss(self, x, z):
        """Loss for Discriminator



        Earth-Mover distance =   ( mean_i ( fw(x(i)) ) − mean_i ( fw(gθ(z(i))) ) (no logarithm)
        Loss = - EM distance = - ( mean_i ( fw(x(i)) ) − mean_i ( fw(gθ(z(i))) )

        Return:
            D_loss: [1]
            D(x): [batch_size, 1]
            D(G(z)): [batch_size, 1]
        """

        # Comput loss with real images
        D_x = self.D(x) # [batch_size, 1]; Discriminator's classification on real images

        # Compute loss with fake images
        G_z = self.G(z) # generate image from noise
        D_G_z = self.D(G_z) # Discriminator's classification on fake images

        D_loss = -(torch.mean(D_x) - torch.mean(D_G_z))

        return D_loss, D_x, D_G_z

    def G_loss(self, z):
        """Loss for Generator
        - mean_i ( fw(gθ(z(i)) )

        Return:
            G_loss: [1]
        """

        # Compute loss with fake images
        G_z = self.G(z) # generate image from noise
        D_G_z = self.D(G_z) # Discriminator's classification on fake images
        G_loss = -torch.mean(D_G_z)

        return G_loss

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

            self.fake_images_path = []
            self.fake_image_tensors = []

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

            self.fake_images_path.append(fake_image_path)

            nrow = int(np.sqrt(batch_size*28*28))
            self.fake_image_tensors.append(fake_images.view(1, nrow, -1))

    def gif(self):
        """Save gif and upload at visdom"""
        images = []

        for path in self.fake_images_path:
            image = imageio.imread(path)
            images.append(image)

        gif_path = os.path.join(self.config.image_log_path, 'generator_growth.gif')
        imageio.mimsave(gif_path, images)

        # (392, 242, 3)
        # imtensor = ndimage.imread(gif_path)

        # (3, 1, 392, 242)
        # imtensor = np.expand_dims(np.rollaxis(imtensor, 2), 1)

        imtensor = torch.stack(self.fake_image_tensors).numpy()
        print(imtensor.shape)
        self.vis.video(tensor=imtensor)

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

        self.D_epoch_loss_history = []
        self.G_epoch_loss_history = []

        for epoch in tqdm(range(self.config.epochs)):
            epoch += 1
            D_epoch_loss = 0
            G_epoch_loss = 0

            data_iter = enumerate(self.data_loader)
            batch_idx = 1
            while batch_idx < self.n_batches_in_epoch:

            # for batch_idx, (images, _) in enumerate(self.data_loader):
                self.D_batch_loss_history = []
                self.G_batch_loss_history = []

                ###########################
                #   Train Discriminator   #
                ###########################

                D_losses = []
                # n_critic = 150 if batch_idx < 25 else self.config.D_n_iter
                n_critic = self.config.D_n_iter
                for _ in range(n_critic):
                    try:
                        # images: [batch_size, 1, 28, 28]
                        batch_idx, (images, _) = next(data_iter)
                    except StopIteration:
                        break

                    batch_idx += 1
                    batch_size = images.size(0)

                    # [batch_size, 1, 28, 28] => [batch_size, 784]
                    x = to_var(images.view(batch_size, -1))

                    # Sampling noise from gaussian distribution
                    z = sample(batch_size, self.config.z_size)

                    # Caculate Loss
                    D_loss, D_x, D_G_z = self.D_loss(x, z)
                    # print('D_loss:', D_loss)
                    # print('G_loss:', G_loss)
                    # print('D(x):', torch.mean(D_x))
                    # print('D(G(z)):', torch.mean(D_G_z))

                    # Backprop & Update
                    self.D.zero_grad()
                    # self.G.zero_grad()
                    D_loss.backward()
                    self.D_optimizer.step()

                    # Weight Clipping (K-Lipschitz condition)
                    self.D_weight_clamp()

                    D_losses.append(D_loss)

                D_loss = np.mean(D_losses)

                #######################
                #   Train Generator   #
                #######################

                # Sampling noise from gaussian distribution
                z = sample(batch_size, self.config.z_size)

                # Calculate Loss
                G_loss = self.G_loss(z)

                # Backprop & Update
                # self.D.zero_grad()
                self.G.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

                # Variable => Tensor => float
                D_loss = to_tensor(D_loss)[0]
                G_loss = to_tensor(G_loss)[0]
                D_x = to_tensor(D_x).mean() # Discriminator's classification among batch
                D_G_z = to_tensor(D_G_z).mean() # Discriminator's classification among batch

                # Batch Loss
                self.D_batch_loss_history.append(D_loss)
                self.G_batch_loss_history.append(G_loss)

                # Print Training details every `log_interval`
                if batch_idx > 1 and batch_idx % self.config.log_interval == 0:
                    progress_percentage = 100. * batch_idx / self.n_batches_in_epoch

                    log_string = f'Epoch [{epoch}/{self.config.epochs}]  Step [{batch_idx}/{self.n_batches_in_epoch}] '
                    log_string += f'({progress_percentage:.0f}%) | '
                    log_string += f'D Loss: {D_loss:.3f} | '
                    log_string += f'G Loss: {G_loss:.3f} | '
                    log_string += f'D(x): {D_x:.3f} | '
                    log_string += f'D(G(z)): {D_G_z:.3f}'

                    print(log_string)

            # Epoch Loss
            D_epoch_loss = np.mean(self.D_batch_loss_history)
            G_epoch_loss = np.mean(self.G_batch_loss_history)
            self.D_epoch_loss_history.append(D_epoch_loss)
            self.G_epoch_loss_history.append(G_epoch_loss)
            print(f'\nEpoch {epoch} | D Loss: {D_epoch_loss:.3f}')
            print(f'Epoch {epoch} | G Loss: {G_epoch_loss:.3f}\n')

            # Plot Learning curve at Visdom
            self.visdom_plot(epoch, D_epoch_loss, G_epoch_loss)

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

        D_loss = 0
        G_loss = 0

        for batch_idx, (images, _) in enumerate(self.data_loader):
            batch_size = images.size(0)

            x = to_var(images.view(batch_size, -1), eval=True)
            z = sample(batch_size, self.config.z_size)

            D_loss, _, _ = self.D_loss(x, z)
            G_loss += self.G_loss(z)


        D_loss /= self.total_data_size
        G_loss /= self.total_data_size

        print(f'D Loss: {D_loss.data[0]:.3f}')
        print(f'G Loss: {G_loss.data[0]:.3f}\n')

    def plot(self):
        """plot learning curve with matplotlib + seaborn
        X: Epoch
        Y: D_loss, G_loss"""

        fig_path = os.path.join(self.config.image_log_path, 'learning_curve.png')
        fig = plt.figure(figsize=(20,8))
        x = range(1, len(self.D_epoch_loss_history)+1)

        with sns.axes_style('whitegrid'):
            plt.plot(x, self.D_epoch_loss_history, label='Discriminator')
            plt.plot(x, self.G_epoch_loss_history, label='Generator')
            plt.title('Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            plt.xlim(1,)
            fig.savefig(fig_path)

    def visdom_plot(self, epoch, D_loss, G_loss):
        """plot learning curve interactively with visdom
        X: Epoch
        Y: D_loss, G_loss"""

        if epoch == 1:
            self.vis = Visdom(env='WGAN')

            # line plot
            self.win = self.vis.line(
                X=np.array([epoch]),
                Y=np.array([[D_loss, G_loss]]),
                opts=dict(
                    title='Learning Curve',
                    xlabel='Epoch',
                    ylabel='Loss',
                    legend=['Discriminator', 'Generator'],
                ))
            # text memo(configuration)
            self.vis.text(str(self.config))

        else:
            self.vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([D_loss]),
                win=self.win,
                name='Discriminator',
            )
            self.vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([G_loss]),
                win=self.win,
                name='Generator',
            )
