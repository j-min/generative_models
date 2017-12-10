import utils
import layers

from model import VAE


class Solver(utils.BaseSolver):
    def build(self):

        self.model = VAE(self.config)

        if self.config.cuda:
            self.model.cuda()
        print(self.model)

        # Build Optimizer (Training Only)
        if self.config.mode == 'train':
            self.optimizer = self.config.optimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, 0.999))

            self.loss_function = layers.VAELoss(self.config.recon_loss)

    def train_step(self, images):
        # Reconstruct Images
        recon_images, mu, log_variance = self.model(images)
        # Calculate loss
        recon_loss, kl_div = self.loss_function(images, recon_images, mu, log_variance)
        return recon_loss, kl_div

    # def eval(self):
    #     print('Start Evaluation!\n')
    #     # Load Parameters from checkpoint
    #     ckpt_path = os.path.join(self.config.save_dir, 'epoch-%d.pkl' % (self.config.epochs))
    #     print('Load parameters from %s' % (ckpt_path))
    #     self.model.load_state_dict(torch.load(ckpt_path))
    #
    #     # set to evaluation mode
    #     self.model.eval()
    #
    #     loss = 0
    #
    #     for batch_idx, (images, _) in enumerate(self.data_loader):
    #         if self.config.cuda:
    #             images = images.cuda()
    #         images = Variable(images, volatile=True)
    #
    #         # Reconstruct Images
    #         recon_images, mu, log_variance = self.model(images)
    #
    #         # Loss
    #         recon_error, kl_div = self.loss_function(recon_images, images, mu, log_variance)
    #         loss += recon_error + kl_div
    #
    #     loss /= self.total_data_size
    #     print(f'Average Test Loss: {loss.data[0]:.4f}\n')
