import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
from tqdm import tqdm
from metrics import denormalize
import metrics
import matplotlib.pyplot as plt
import warnings
from config import mean_value, comp_idx, in_channels
warnings.filterwarnings("ignore")



class Trainer_VSR:
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        print("Running on device: ", self.device)
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp
        self.loss = nn.MSELoss() #

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step()

    def set_loader(self, new_loader):
        self.loader_train = new_loader.loader_train
        self.loader_test = new_loader.loader_test

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        print("VSR training")
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1

        lr = self.scheduler.get_last_lr()[0]

        self.model.train()
        self.ckp.start_log()
        for batch, (lr, hr) in enumerate(self.loader_train):

            # divide LR frame sequence [N, n_sequence, n_colors, H, W] -> n_sequence * [N, 1, n_colors, H, W]
            #lr = list(torch.split(lr, self.args.n_colors, dim = 1))
            lr = torch.chunk(lr, self.args.n_sequence, dim=1)

            # target frame = middle HR frame [N, n_colors, H, W]
            hr = hr[:, int(hr.shape[1]/2), : ,: ,:]

            lr = [x.to(self.device) for x in lr]
            hr = hr.to(self.device)

            self.optimizer.zero_grad()

            # output frame = single HR frame [N, n_colors, H, W]
            if self.model.get_model().name == 'ESPCN_mf':
                sr = self.model(lr).to(self.device) # added to device argument
                loss = self.loss(sr, hr)

            else: 
               print("Wrong model selected.")

            self.ckp.report_log(loss.item())
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            if (batch + 1) == len(self.loader_train):
              self.ckp.write_log('Epoch {} (train) completed. Average train MSE Loss: {:.5f}'.format(
              epoch, self.ckp.loss_log[-1] / len(self.loader_train)))

        self.ckp.end_log(len(self.loader_train))


    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)

        # collect sr images
        sr_images = []

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (lr, hr) in enumerate(tqdm_test):
                filename = "super_images.frame"
                

                # extract middle frame
                hr = hr[:, int(hr.shape[1]/2), :, :] 

                # lr images
                #lr = list(torch.split(lr, self.args.n_colors, dim = 1))
                lr = torch.chunk(lr, self.args.n_sequence, dim=1)


                lr = [x.to(self.device) for x in lr]
                hr = hr.to(self.device)

                # output frame = single HR frame [N, n_colors, H, W]
                if self.model.get_model().name == 'ESPCN_mf':
                    sr = self.model(lr)

                else: 
                    print("Wrong model selected.")
                
                # de-normalize images
                hr, sr, lr = denormalize(hr, comp_idx, mean=mean_value), denormalize(sr, comp_idx, mean=mean_value), denormalize(lr[1], comp_idx, mean=mean_value)

                if in_channels == 2: 
                   hr = hr[:, comp_idx, :, :]
                   sr = sr[:, comp_idx, :, :]
                
                else: 
                   pass 

                
                # save sr images as .pt 
                sr_img = sr.cpu().squeeze()
                sr_images.append(sr_img)
            
                hr = hr.to(self.device)  
                sr = sr.to(self.device)

                '''
            
                sr_images_tensor = torch.stack(sr_images).view(-1, 32, 32)
                torch.save(sr_images_tensor, f'experiment/sr_mf_espcn_{comp_idx}_{self.args.scale}.pt')
            
                ''' 
                
                PSNR = metrics.calc_psnr(sr, hr)
                MAE =  metrics.calc_mae(sr, hr)
                SSIM = metrics.calc_ssim(sr, hr)
                MSE = metrics.calc_mse(sr, hr)
                
                self.ckp.report_log(PSNR, train=False)
                self.ckp.report_mae(MAE)
                self.ckp.report_ssim(SSIM)
                self.ckp.report_mse(MSE)


            if self.args.save_images:
              save_list = [lr, hr, sr]
              self.ckp.save_images(filename, save_list, self.args.scale)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.mae_log.min(0)

            best_psnr = self.ckp.psnr_log.max(0)

            self.ckp.write_log('[{}]\taverage PSNR: {:.5f} (Best: {:.5f} @epoch {})'.format(
                                self.args.data_test, self.ckp.psnr_log[-1],
                                best_psnr[0], best_psnr[1] + 1))


            if self.ckp.mae_log.numel() > 0:
              last_mae, last_mae_index = self.ckp.mae_log.min(0)
        

              self.ckp.write_log('[{}]\taverage MAE: {:.5f} (Best: {:.5f} @epoch {})'.format(
                        self.args.data_test, self.ckp.mae_log[-1],
                        last_mae, last_mae_index.item() + 1))

            else:
              last_mae = None
              self.ckp.write_log('[{}]\tMAE log is empty.'.format(self.args.data_test))


            if self.ckp.mse_log.numel() > 0:
                last_mse, last_mse_index = self.ckp.mse_log.min(0)

                self.ckp.write_log('[{}]\taverage MSE: {:.5f} (Best: {:.5f} @epoch {})'.format(
                            self.args.data_test, self.ckp.mse_log[-1],
                            last_mse, last_mse_index.item() + 1))

            else:
                last_mae = None
                self.ckp.write_log('[{}]\tMAE log is empty.'.format(self.args.data_test))

            if self.ckp.ssim_log.numel() > 0:
              last_ssim, last_ssim_index = self.ckp.ssim_log.max(0)


              self.ckp.write_log('[{}]\taverage SSIM: {:.5f} (Best: {:.5f} @epoch {})'.format(
                        self.args.data_test, self.ckp.ssim_log[-1],
                        last_ssim, last_ssim_index.item() + 1))

            else:
              last_ssim = None
              self.ckp.write_log('[{}]\tSSIM log is empty.'.format(self.args.data_test))



            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
        


    def plot_images(self, num_images=1): 
       self.model.eval()
       with torch.no_grad(): 
          for idx_img, (lr, hr) in enumerate(self.loader_test): 
             if idx_img >= num_images: 
                break
             
             if in_channels ==2:
                lr = torch.chunk(lr, self.args.n_sequence, dim=1)
             else: 
                pass
             
             lr = [x.to(self.device) for x in lr]

          
             sr = self.model(lr)

             if in_channels == 1: 
                lr = lr[0][1, :, :, :] # 1: extract middle frame
                hr = hr[:, 1, :, :, :]

             else: 
                lr = lr[0][1, comp_idx, :, :] # plot component: comp_idx
                hr = hr[: ,1, comp_idx, :, :]
                sr = sr[comp_idx, :, :]
             
             hr = denormalize(hr, comp_idx, mean=mean_value).to(self.device)
             sr = denormalize(sr, comp_idx, mean=mean_value).to(self.device)
             lr = denormalize(lr, comp_idx, mean=mean_value).to(self.device)

             hr = hr.squeeze().cpu().numpy()
             sr = sr.squeeze().cpu().numpy()
             lr = lr.squeeze().cpu().numpy()

             plt.figure(figsize=(12, 4))
             plt.subplot(1, 3, 1)
             plt.imshow(lr)
             plt.title('LR Input Image')

             plt.subplot(1, 3, 2)
             plt.imshow(hr)
             plt.title('HR Ground Truth Image')

             plt.subplot(1, 3, 3)
             plt.imshow(sr)
             plt.title('HR Prediction Image')
             plt.savefig(os.path.join('src/models/mf_espcn/VESPCN-PyTorch/experiment/save_path', f'results_{comp_idx}_{self.args.scale}.png'))



    def terminate(self):
      if self.args.test_only:
        self.test()
        return True
      else:
        epoch = self.scheduler.last_epoch + 1
        return epoch >= self.args.epochs
      

      