import torch
import numpy as np
from tqdm import tqdm
from .image import save_image
# from .convert import to_var
from torch.autograd import Variable
import layers


class BaseSolver(object):
    def __init__(self, config, train_loader, is_train=False):
        """Base Solver Class for VAE"""
        self.config = config
        self.train_loader = train_loader
        self.n_batches_in_epoch = len(self.train_loader)
        self.total_data_size = len(self.train_loader.dataset)
        self.is_train = self.config.isTrain

        self.batch_loss_history = []
        self.batch_recon_loss_history = []
        self.batch_kl_div_history = []

        self.epoch_loss_history = []
        self.epoch_recon_loss_history = []
        self.epoch_kl_div_history = []

    def build(self):
        """
        Train: Model / Loss / Optimizer
        Eval: Model
        """
        raise NotImplementedError

    def loss_function(self):
        """Calculate loss"""
        raise NotImplementedError

    def initialize(self):
        """Initialize Model Parameters (Default: Xavier)"""
        # layers.init_weights(self.model, self.config.init_weights)

    def save_image(self, images, epoch_i, path):
        """Save images"""
        recon_image_path = self.config.ckpt_dir.joinpath(path)
        if not self.config.ckpt_dir.exists():
            self.config.ckpt_dir.mkdir()
        print(f'Save {recon_image_path}')
        save_image(images.data, recon_image_path)

    def save(self, epoch_i, network=None):
        """Save parameters at checkpoint"""
        if network is None:
            network = self.model
        if not self.config.ckpt_dir.exists():
            self.config.ckpt_dir.mkdir()
        ckpt_path = self.config.ckpt_dir.joinpath(f'epoch-{epoch_i}.pkl')
        print(f'Save parameters at {ckpt_path}')
        torch.save(network.cpu().state_dict(), ckpt_path)

    def load(self, epoch_i, network=None):
        """Load Parameters from checkpoint"""
        if network is None:
            network = self.model
        ckpt_path = self.config.ckpt_dir.join_path(f'epoch-{epoch_i}.pkl')
        print(f'Load parameters from {ckpt_path}')
        network.load_state_dict(torch.load(ckpt_path))

    def train_step(self):
        raise NotImplementedError

    def train(self):
        print('Start Training!\n')
        # Initialize parameteres
        self.initialize()

        for epoch_i in tqdm(range(self.config.n_epochs)):
            epoch_i += 1

            self.model.train()

            for batch_i, (images, _) in enumerate(self.train_loader):
                batch_i += 1

                batch_size, n_channel, height, width = images.size()

                if self.config.layer == 'mlp':
                    images = images.view(batch_size, -1)

                images = Variable(images)  # [batch_size, 1, 28, 28]
                if self.config.cuda:
                    images = images.cuda()
                # import ipdb
                # ipdb.set_trace()

                # Forward Propagation & Calculate Loss
                recon_loss, kl_div = self.train_step(images)
                batch_loss = recon_loss + kl_div

                # Backpropagation
                self.optimizer.zero_grad()
                batch_loss.backward()

                # Update Parameters
                self.optimizer.step()

                recon_loss = float(recon_loss.data)
                kl_div = float(kl_div.data)
                batch_loss = float(batch_loss.data)
                self.batch_recon_loss_history.append(recon_loss)
                self.batch_kl_div_history.append(kl_div)
                self.batch_loss_history.append(batch_loss)

                if batch_i > 1 and batch_i % self.config.log_interval == 0:
                    progress_percentage = 100. * batch_i / self.n_batches_in_epoch

                    log_string = f'Epoch [{epoch_i}/{self.config.n_epochs}]  Step [{batch_i}/{self.n_batches_in_epoch}] '
                    log_string += f'({progress_percentage:.0f}%) | '
                    log_string += f'Batch Loss: {batch_loss/batch_size:.2f} | '
                    log_string += f'Reconstruction Loss: {recon_loss/batch_size:.2f} | '
                    log_string += f'KL-Divergence: {kl_div/batch_size:.2f}'
                    print(log_string)

            epoch_recon_loss = np.mean(self.batch_recon_loss_history)
            epoch_kl_div = np.mean(self.batch_kl_div_history)
            epoch_loss = np.mean(self.batch_loss_history)

            self.epoch_recon_loss_history.append(epoch_recon_loss)
            self.epoch_kl_div_history.append(epoch_kl_div)
            self.epoch_loss_history.append(epoch_loss)

            log_string = f'Epoch {epoch_i} | '
            log_string += f'Loss: {epoch_loss:.2f} | '
            log_string += f'Reconstruction Loss: {epoch_recon_loss:.2f} | '
            log_string += f'KL-Divergence: {epoch_kl_div:.2f}'
            print(log_string)

            self.model.eval()

            # Save original images
            if epoch_i == 1:
                fixed_images = images  # for debugging
                self.save_image(images, epoch_i, 'original_images.png')

            # Save reconstructed images
            recon_images, _, _ = self.model(fixed_images)
            self.save_image(recon_images, epoch_i, f'recon_images-{epoch_i}.png')

            # Save parameters at checkpoint
            self.save(epoch_i)

    # def train(self):
    #     """
    #     Train model
    #
    #     self.initialize()
    #     self.model.train()
    #
    #     for epoch_i in tqdm(range(self.config.epochs)):
    #         ...
    #         for step_i, (images, _) in enumerate(self.train_loader):
    #             ...
    #             model()
    #             loss = loss_fn()
    #             loss.backward()
    #             optimizer.step()
    #             ...
    #     """
    #     raise NotImplementedError

    def eval(self):
        """
        Evaluate model

        self.initialize()
        self.model.eval()

        for epoch_i in tqdm(range(self.config.epochs)):
            ...
            for step_i, (images, _) in enumerate(self.test_loader):
                ...
                model()
                loss = loss_fn()
                loss.backward()
                optimizer.step()
                ...
        """
        raise NotImplementedError
