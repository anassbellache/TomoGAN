import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from .unet_parts import OutConv, DoubleConv, Down, Up
from .base_model import BaseModel
from .loss_functions import GANLoss, PixelLoss
import torchvision.models as models
from utils.image_utils import save2image



def calculate_feature_output_size(img_size, kernel_size, padding, stride):
    return int((img_size - kernel_size + 2*padding)/stride) + 1


class UNet(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UNet, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.inc = OutConv(channels_in, out_channels=8, relu=True)
        self.conv1 = DoubleConv(8, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 128)
        self.bottom = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1),
            nn.ReLU()
        )

        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 32)
        self.conv2 = OutConv(32, 16, relu=True)
        self.conv3 = OutConv(16, channels_out, relu=False)
    
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.conv1(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.bottom(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

        

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        N = calculate_feature_output_size(img_size, 3, 0, 1)
        N = calculate_feature_output_size(N, 3, 0, 2)
        N = calculate_feature_output_size(N, 3, 0, 1)
        N = calculate_feature_output_size(N, 3, 0, 2)
        N = calculate_feature_output_size(N, 3, 0, 1)
        self.feature_size = calculate_feature_output_size(N, 3, 0, 2)
        self.image_to_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=2),
            nn.LeakyReLU()
        )
        self.features_to_score = nn.Sequential(
            nn.Linear(4*self.feature_size*self.feature_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1))
    
    def forward(self, x):
        x = self.image_to_features(x)
        x = x.contiguous().view((x.shape[0], -1))
        x = self.features_to_score(x)
        return x

class TomoGAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_B', 'fake_C', 'real_C']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = UNet(opt.depth, 1)
        if self.isTrain:
            self.netD = Discriminator(opt.image_size)
            self.netD.to(self.device)
     
        self.netG.to(self.device)
        
        self.itr_out_dir = opt.name + '-itrOut'
        
        self.depth = opt.depth

        if self.isTrain:
            vgg19_dict = torch.load(opt.vgg_path)
            vgg19 = models.vgg19(pretrained=False)
            vgg19.load_state_dict(vgg19_dict)
            if torch.cuda.is_available():
                vgg19.features.cuda()
            self.criterionGAN = GANLoss(self.device, gan_mode='vanilla')
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionPixel = PixelLoss(vgg19.features, self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if torch.cuda.is_available():
                vgg19.features.cuda()
        else:
            web_dir = os.path.join(opt.results_dir, opt.name, str(opt.load_epoch))
            self.image_paths = ['{}/images/{}.png'.format(web_dir, 'reconstructed'),
                            '{}/images/gtruth.png'.format(web_dir),
                            '{}/images/blurred.png'.format(web_dir)]
    
    def set_input(self, input):
        X_mb, y_mb = input[0], input[1]
        #X_mb = np.transpose(X_mb, (0, 3, 1, 2))
        #y_mb = np.transpose(y_mb, (0, 3, 1, 2))
        self.real_B = torch.tensor(X_mb, dtype=torch.float32, requires_grad=True).to(self.device)
        self.real_C = torch.tensor(y_mb, dtype=torch.float32, requires_grad=True).to(self.device)
        
    
    def forward(self):
        self.fake_C = self.netG(self.real_B)
    
    def compute_visuals(self):
        self.fake_C = self.fake_C[0,0,:,:]
        self.real_C = self.real_C[0,0,:,:]
        self.real_B = self.real_B[0,0:self.depth//2,:,:].squeeze(0)
        #save2image(self.fake_C[0,0,:,:].detach().cpu().numpy(), self.image_paths[0])
        #save2image(self.real_C[0,0,:,:].detach().cpu().numpy(), self.image_paths[1])
        #save2image(self.real_B[0,0:self.depth//2,:,:], self.image_paths[2])
        
    
    def backward_D(self):
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netG, False)
        self.optimizer_D.zero_grad()
        gen_C = self.netG(self.real_B)
        pred_fake = self.netD(gen_C)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netD(self.real_C)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()
        self.optimizer_D.step()
        for p in self.netD.parameters():
            p.data.clamp_(-0.01, 0.01)
    
    def backward_G(self):
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        gen_C = self.netG(self.real_B)
        pred_fake = self.netD(gen_C)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_MSE = self.criterionMSE(gen_C, self.real_C) * self.opt.lmse
        self.loss_G_Perc = self.criterionPixel(gen_C, self.real_C) * self.opt.lperc
        self.loss_G = self.loss_G_GAN + self.loss_G_MSE + self.loss_G_Perc
        self.loss_G.backward()
        self.optimizer_G.step()
    
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    

if __name__ == "__main__":
    batch_size = 3
    pictures = torch.randint(0, 256, (batch_size, 1024, 1024))
    print(pictures.shape)
    net_G = UNet(batch_size, 1)
    generated = net_G(pictures)
    #print(generated.shape) 
    
