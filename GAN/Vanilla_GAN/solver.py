import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torchvision.utils import save_image

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

        # Binary Cross-Entropy
        self.BCE = nn.BCELoss(size_average=False) # losses are summed

        # Build Optimizer (Training Only)
        if self.is_train:
            self.d_optimizer = self.config.optimizer(
                self.D.parameters(),
                lr=self.config.d_learning_rate,
                betas=(self.config.beta1, self.config.beta2))
            self.g_optimizer = self.config.optimizer(
                self.G.parameters(),
                lr=self.config.g_learning_rate,
                betas=(self.config.beta1, self.config.beta2))

    def d_loss(self, x, z, batch_size):
        """Loss for Discriminator
        Binary Cross-Entropy(Discriminator(real_images), All_real)
        + Binary Cross-Entropy(1 - Discriminator(fake_images), All_fake)
        = - ( sum_i(log(D(x_i)) +       log(1 − D(G(z_i))) )
        = -   sum_i(log(D(x_i)) - sum_i(log(1 − D(G(z_i))))

        Return:
            d_loss_real: [1]
            d_loss_fake: [1]
            D(x): [batch_size, 1]
            D(G(z)): [batch_size, 1]
        """
        all_ones = to_var(torch.ones(batch_size))
        all_zeros = to_var(torch.zeros(batch_size))

        # Comput loss with real images
        D_x = self.D(x) # [batch_size, 1]; Discriminator's classification on real images
        d_loss_real = self.BCE(D_x, all_ones) # BCE: Discriminator's classification vs all 1s

        # Compute loss with fake images
        G_z = self.G(z) # generate image from noise
        D_G_z = self.D(G_z) # Discriminator's classification on fake images
        d_loss_fake = self.BCE(D_G_z, all_zeros) # BCE: Discriminator's classification vs all 0s

        d_loss = d_loss_real + d_loss_fake

        return d_loss, D_x, D_G_z

    def g_loss(self, z, batch_size):
        """Loss for Generator
        Original Loss: sum_i(log(1 - D(G(z_i))))
        Modified Loss: sum_i(log(    D(G(z_i))))

        Return:
            g_loss: [1]
        """
        all_ones = to_var(torch.ones(batch_size))

        # Compute loss with fake images
        G_z = self.G(z) # generate image from noise
        D_G_z = self.D(G_z) # Discriminator's classification on fake images
        g_loss = self.BCE(D_G_z, all_ones) # KL between Discriminator's classification and all 1s

        return g_loss

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

        # Initialize parameteres
        # initialize(self.D)
        initialize(self.G)

        # Set to training mode
        self.D.train()
        self.G.train()

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

                # Sampling noise from gaussian distribution
                z = sample(batch_size, self.config.z_size)

                # Caculate Loss
                d_loss, D_x, D_G_z = self.d_loss(x, z, batch_size)

                # Backprop & Update
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                #######################
                #   Train Generator   #
                #######################

                # Sampling noise from gaussian distribution
                z = sample(batch_size, self.config.z_size)

                # Calculate Loss
                g_loss = self.g_loss(z, batch_size)

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

            # Save real images
            if epoch == 1:
                fixed_images = x # for debugging
                fixed_noise = z # for debugging

                images = x.view(batch_size, 1, 28, 28)
                real_image_path = os.path.join(self.config.image_log_path, 'real_images.png')
                print('Save real images at %s' % (real_image_path))
                save_image(images.data, real_image_path)

            if epoch % self.config.save_interval == 0:
                # Save fake images
                fake_images = self.G(fixed_noise)
                fake_images = fake_images.view(batch_size, 1, 28, 28).data
                # fake_images = denorm(fake_images.data) # (-1, 1) => (0, 1)
                fake_image_path = os.path.join(self.config.image_log_path, 'fake_images-%d.png' % (epoch))
                print('Save fake images at %s' % (fake_image_path))
                save_image(fake_images, fake_image_path)

                # Save parameters at checkpoint
                d_ckpt_path = os.path.join(self.config.save_dir, 'D_epoch-%d.pkl' % (epoch))
                print('Save parameters at %s' % (d_ckpt_path))
                torch.save(self.D.state_dict(), d_ckpt_path)

                g_ckpt_path = os.path.join(self.config.save_dir, 'G_epoch-%d.pkl' % (epoch))
                print('Save parameters at %s\n' % (g_ckpt_path))
                torch.save(self.G.state_dict(), g_ckpt_path)

    def eval(self):
        print('Start Evaluation!\n')
        # Load Parameters from checkpoint
        d_ckpt_path = os.path.join(self.config.save_dir, 'D_epoch-%d.pkl' % (self.config.epochs))
        print('Load parameters from %s' % (d_ckpt_path))
        self.D.load_state_dict(torch.load(d_ckpt_path))

        g_ckpt_path = os.path.join(self.config.save_dir, 'G_epoch-%d.pkl' % (self.config.epochs))
        print('Load parameters from %s' % (g_ckpt_path))
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

            d_loss, _, _ = self.d_loss(x, z, batch_size)
            g_loss += self.g_loss(z, batch_size)


        d_loss /= self.total_data_size
        g_loss /= self.total_data_size

        print(f'D Loss: {d_loss.data[0]:.3f}')
        print(f'G Loss: {g_loss.data[0]:.3f}\n')
