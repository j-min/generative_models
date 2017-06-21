import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

import os
from tqdm import tqdm
from model import VAE

class Solver(object):
    def __init__(self, config, data_loader, is_train=False):
        self.config = config
        self.data_loader = data_loader
        self.n_batches_in_epoch = len(self.data_loader)
        self.total_data_size = len(self.data_loader.dataset)
        self.is_train = is_train

    def build_model(self):
        self.model = VAE(self.config)

        if self.config.cuda:
            model.cuda()

        # Binary Cross-Entropy (Reconstruction Error)
        self.binary_cross_entropy = nn.BCELoss()
        self.binary_cross_entropy.size_average = False # losses are summed

        # Build Optimizer (Training Only)
        if self.is_train:
            self.optimizer = self.config.optimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2))

    def loss_function(self, recon_x, x, mu, log_variance):
        """Reconstruction error + Regularizer
        (= Binary cross-entropy - KL-Divergence)
        KL = 0.5 * sum(1+log(sigma^2) - mu^2 - sigma^2)"""

        reconstruction_error = self.binary_cross_entropy(recon_x, x)

        KL_divergence = mu.pow(2).add_(log_variance.exp()).mul_(-1).add_(1).add_(log_variance)
        KL_divergence = torch.sum(KL_divergence).mul_(-0.5)

        return reconstruction_error, KL_divergence

    def denorm(self, x):
        """(-1, 1) => (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def initialize(self):
        """Apply Xavier initilization on weight parameters"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal(module.weight.data)

    def train(self):
        print('Start Training!\n')
        # Initialize parameteres
        self.initialize()

        # Set to training mode
        self.model.train()

        for epoch in tqdm(range(self.config.epochs)):
            epoch += 1
            epoch_loss = 0
            for batch_idx, (images, _) in enumerate(self.data_loader):
                batch_idx += 1
                images = Variable(images) # [batch_size, 1, 28, 28]
                if self.config.cuda:
                    images = images.cuda()
                self.optimizer.zero_grad()

                # Reconstruct Images
                recon_images, mu, log_variance = self.model(images)

                # Loss
                recon_error, kl_div = self.loss_function(recon_images, images, mu, log_variance)
                batch_loss = recon_error + kl_div
                epoch_loss += batch_loss

                # Backpropagation
                batch_loss.backward()

                # Update Parameters
                self.optimizer.step()

                if batch_idx > 1 and batch_idx % self.config.log_interval == 0:
                    progress_percentage = 100. * batch_idx / self.n_batches_in_epoch
                    average_batch_loss = batch_loss.data[0] / len(images)
                    average_recon_error = recon_error.data[0] / len(images)
                    average_kl_div = kl_div.data[0] / len(images)

                    log_string = f'Epoch [{epoch}/{self.config.epochs}]  Step [{batch_idx}/{self.n_batches_in_epoch}] '
                    log_string += f'({progress_percentage:.0f}%) | '
                    log_string += f'Average Batch Loss: {average_batch_loss:.2f} '
                    log_string += f'Reconstruction Error: {average_recon_error:.2f} '
                    log_string += f'KL-Divergence: {average_kl_div:.2f}'

                    print(log_string)

            average_epoch_loss = epoch_loss.data[0] / self.total_data_size
            print(f'Epoch {epoch} | Average Epoch Loss: {average_epoch_loss:.2f}\n')

            # Save original images
            if epoch == 1:
                fixed_images = images # for debugging
                images = images.view(images.size(0), 1, 28, 28)
                original_image_path = os.path.join(self.config.image_log_path, 'original_images.png')
                print('Save original images at %s' % (original_image_path))
                save_image(images.data, original_image_path)

            # Save reconstructed images
            recon_images, _, _ = self.model(fixed_images)
            recon_images = recon_images.view(recon_images.size(0), 1, 28, 28)
            recon_image_path = os.path.join(self.config.image_log_path, 'recon_images-%d.png' % (epoch))
            print('Save reconstructed images at %s' % (recon_image_path))
            save_image(recon_images.data, recon_image_path)

            # Save parameters at checkpoint
            ckpt_path = os.path.join(self.config.save_dir, 'epoch-%d.pkl' % (epoch))
            print('Save parameters at %s' % (ckpt_path))
            torch.save(self.model.state_dict(), ckpt_path)

    def eval(self):
        print('Start Evaluation!\n')
        # Load Parameters from checkpoint
        ckpt_path = os.path.join(self.config.save_dir, 'epoch-%d.pkl' % (self.config.epochs))
        print('Load parameters from %s' % (ckpt_path))
        self.model.load_state_dict(torch.load(ckpt_path))

        # set to evaluation mode
        self.model.eval()

        loss = 0

        for batch_idx, (images, _) in enumerate(self.data_loader):
            if self.config.cuda:
                images = images.cuda()
            images = Variable(images, volatile=True)

            # Reconstruct Images
            recon_images, mu, log_variance = self.model(images)

            # Loss
            recon_error, kl_div = self.loss_function(recon_images, images, mu, log_variance)
            loss += recon_error + kl_div

        loss /= self.total_data_size
        print(f'Average Test Loss: {loss.data[0]:.4f}\n')
