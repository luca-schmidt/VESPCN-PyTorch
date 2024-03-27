import torch
import imageio
import numpy as np
import os
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from config import comp_idx, in_channels


class Logger:
    def __init__(self, args):
        self.args = args
        self.psnr_log = torch.Tensor()
        self.loss_log = torch.Tensor()
        self.mae_log  = torch.Tensor() 
        self.mse_log =  torch.Tensor()
        self.ssim_log = torch.Tensor() 
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # added this line of code

        if args.load == '.':
            if args.save == '.':
                args.save = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
            self.dir = 'experiment/' + args.save
        else:
            self.dir = 'experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.loss_log = torch.load(self.dir + '/loss_log.pt')
                self.psnr_log = torch.load(self.dir + '/psnr_log.pt')
                self.mae_log = torch.load(self.dir + '/mae_log.pt') 
                self.mse_log = torch.load(self.dir + '/mse_log.pt')
                self.ssim_log = torch.load(self.dir + '/ssim_log.pt')
                print('Continue from epoch {}...'.format(len(self.psnr_log)))

        if args.reset:
            os.system('rm -rf {}'.format(self.dir))
            args.load = '.'

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            if not os.path.exists(self.dir + '/model'):
                os.makedirs(self.dir + '/model')
        if not os.path.exists(self.dir + '/result/'+self.args.data_test):
            print("Creating dir for saving images...", self.dir + '/result/'+self.args.data_test)
            os.makedirs(self.dir + '/result/'+self.args.data_test)

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

    def save(self, trainer, epoch, is_best):
        trainer.model.save(self.dir, is_best)
        #torch.save(self.loss_log, os.path.join(self.dir, 'loss_log.pt'))
        #torch.save(self.psnr_log, os.path.join(self.dir, 'psnr_log.pt'))
        #torch.save(self.mae_log, os.path.join(self.dir, 'mae_log.pt')) 
        #torch.save(self.mse_log, os.path.join(self.dir, 'mse_log.pt'))
        #torch.save(self.ssim_log, os.path.join(self.dir, 'ssim_log.pt'))
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
        self.plot_loss_log(epoch)
        self.plot_psnr_log(epoch)


    def save_images(self, filename, save_list, scale):
        if self.args.task == 'Image':
            filename = '{}/result/{}/{}_x{}_'.format(self.dir, self.args.data_test, filename, scale)
            postfix = ['LR', 'HR', 'SR']

        elif self.args.task == 'Video':
            f = filename.split('.')
            filename = '{}/result/{}/{}/{}_'.format(self.dir, self.args.data_test, f[0], f[1])
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            postfix = ['LR', 'HR', 'SR']

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

        for img, post, ax in zip(save_list, postfix, axs):
            img = img[0].data  # .mul(255 / self.args.rgb_range)

            if len(img.shape) == 5 and img.shape[2] == 2 and in_channels == 2:
                img = img[:, :, comp_idx, :, :]
            else:
                pass

            if len(img.shape) == 5:
                img = img.squeeze(0).squeeze(0).squeeze(0)

            if len(img.shape) == 4:
                img = img[0, 0, :, :]

            if len(img.shape) == 3:
                img = img.squeeze(0)

            # bring to device
            img = img.detach().cpu()

            ax.imshow(img, cmap='viridis')
            ax.set_title(post)  # Set title for each subplot

        # Save the entire figure
        #save_path = '{}combined_plot.png'.format(filename)
        #plt.savefig('{}_plot_new.png'.format(self.dir))
        plt.show()


    def start_log(self, train=True):
        if train:
            self.loss_log = torch.cat((self.loss_log, torch.zeros(1)))
        else:
            self.mae_log  = torch.cat((self.mae_log, torch.zeros(1)))
            self.psnr_log = torch.cat((self.psnr_log, torch.zeros(1)))
            self.ssim_log = torch.cat((self.ssim_log, torch.zeros(1)))
            self.mse_log  = torch.cat((self.mse_log, torch.zeros(1)))


    def report_log(self, item, train=True):
        if train:
            self.loss_log[-1] += item

        else:
            if item.device.type == 'cuda': 
              item = item.detach().cpu()
            self.psnr_log[-1] += item


    def report_mae(self, item):
      item = item.detach().cpu() 
      self.mae_log[-1] += item

    def report_mse(self, item): 
        item = item.detach().cpu()
        self.mse_log[-1] += item

    def report_ssim(self, item):
      self.ssim_log[-1] += item


    def end_log(self, n_div, train=True):
        if train:
            self.loss_log[-1].div_(n_div)
        else:
            self.mae_log[-1].div_(n_div)
            self.mse_log[-1].div_(n_div)
            self.psnr_log[-1].div_(n_div)
            self.ssim_log[-1].div_(n_div)



    def plot_loss_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)

        # inserted modification
        if len(self.loss_log.shape) > 1:
          self.loss_log = self.loss_log.squeeze()

        if len(axis) != len(self.loss_log):
          axis = np.arange(1, len(self.loss_log) + 1)

        #print("Axis shape:", axis.shape)
        #print("Loss log shape:", self.loss_log.shape)

        fig = plt.figure()
        plt.title('Loss Graph')
        plt.plot(axis, self.loss_log.numpy())
        #plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'loss.pdf'))
        plt.close(fig)

    def plot_psnr_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')

        # inserted modificatioin
        if len(self.psnr_log.shape) > 1:
          self.psnr_log = self.psnr_log.squeeze()

        if len(axis) != len(self.psnr_log):
          axis = np.arange(1, len(self.psnr_log) + 1)

        plt.plot(axis, self.psnr_log.numpy())
        #plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'psnr.pdf'))
        plt.close(fig)

    def done(self):
        self.log_file.close()