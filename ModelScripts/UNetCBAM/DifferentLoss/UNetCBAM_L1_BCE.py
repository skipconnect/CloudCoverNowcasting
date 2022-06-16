# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.transforms import CenterCrop

import os

import argparse
from argparse import Namespace

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for ConvLSTM layers')
parser.add_argument('--input_dim', type=int, default=3, help='Input dimension for first ConvLSTM layer')
parser.add_argument('--frame_size', type=int, default=128, help='Input frame size as integer (assumed quadratic), after downsampling')
parser.add_argument('--input_seq', type=int, default=4, help='Input sequence length')
parser.add_argument('--output_seq', type=int, default=6, help='Output sequence length')
parser.add_argument('--version', type=int, default=1, help='Version 1 is with all 6 frames in one, version 2 is with lead_time')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--model_description', type=str, help='Description of the model', required=True)
parser.add_argument('--auto_lr', default=False, type=bool, help = "choose to autofind lr")
parser.add_argument('--only_cloud', default=False, type=bool, help = "Choose to use only cloud mask or additinal data")
parser.add_argument('--continue_from_latest', default=False, type=bool, help = "Continue from latest checkpoint")
parser.add_argument('--testing', default=False, type=bool, help = "Testting or no testing")
parser.add_argument('--leadtime', type=int, default=0, help='Lead time for version2')
parser.add_argument('--gradbatches', type=int, default=1, help='Number of batches to accumululate before calculating gradients')
parser.add_argument('--lambda_BCE', type=float, default=1, help='The amount to weight the BCE loss')
parser.add_argument('--lambda_L1', type=float, default=1, help='The amount to weight the L1 loss')

opt = parser.parse_args()




# Loading data from folder and files (if statements to control the right input_dim for the different version and if we only use cloud mask as input)
if opt.only_cloud:
    folder_dir_ch="./data/16bit_1dim"
    chunks=sorted(os.listdir(folder_dir_ch))
    final_list = []
    chunklist=[]
    seqlen=opt.input_seq + opt.output_seq
    window=3
    j=0
    print("Loading data in to Main Memory (Only-Cloud)")
    for chunk in chunks[1:]:
        file=torch.load(folder_dir_ch +"/" + chunk)
        for i in range(0,file.shape[0], window):
            if i<file.shape[0]-seqlen:
                chunklist.append(file[i:seqlen+i,:,:,:].transpose(0,1).type(torch.uint8))
            else:
                break
    print("\nDone")
    # Overwriting input_dim corresponding to the version
    if opt.version == 1:
        opt.input_dim = 1
    else:
        opt.input_dim = 2
else:
    folder_dir_ch="./data/16bitdone"
    chunks=sorted(os.listdir(folder_dir_ch))
    final_list = []
    chunklist=[]
    seqlen=opt.input_seq + opt.output_seq
    window=3
    j=0
    print("Loading data in to Main Memory (Full-Data)")
    for chunk in chunks[1:]:
        file=torch.load(folder_dir_ch +"/" + chunk)
        for i in range(0,file.shape[0], window):
            if i<file.shape[0]-seqlen:
                chunklist.append(file[i:seqlen+i,:,:,:].transpose(0,1))
            else:
                break
    print("\nDone")
    # Overwriting input_dim corresponding to the version
    if opt.version == 1:
        opt.input_dim = 3
    else:
        opt.input_dim = 4 

##############################################################################
# Defining datasets

#Setting a random.seed (this is for the shuffeling of the dataset, to ensure it is the same across each run)
np.random.seed(12)
datatype = torch.float16
if opt.version == 1: #(With all 6 frames at once)
    class mydata(Dataset):
        def __init__(self, datalist, input_seq):
            self.data= datalist
            self.input_seq = input_seq
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            batch = self.data[idx]
            input_ = batch[:,:self.input_seq,]
            target = batch[0,self.input_seq:,64:192, 64:192]
            #Reshaping inpu-size by collapsing channels,input_seq to one dimension of channels*input_seq (concatanating)
            input_ = input_.reshape((input_.shape[0]*input_.shape[1],input_.shape[2],input_.shape[3]))
            return input_.type(datatype), target.type(datatype)
            
    class mydataTest(Dataset):
        def __init__(self, datalist, input_seq):
            self.data= datalist
            self.input_seq = input_seq
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            batch = self.data[idx]
            input_ = batch[:,:self.input_seq,]
            target = batch[0,self.input_seq:,64:192, 64:192]
            #Reshaping inpu-size by collapsing channels,input_seq to one dimension of channels*input_seq (concatanating)
            input_ = input_.reshape((input_.shape[0]*input_.shape[1],input_.shape[2],input_.shape[3]))
            return input_.type(datatype), target.type(datatype)


if opt.version == 2: #(With lead-time)
    # Overwriting output_seq (if version is 2): Since we are only predicting one frame we set output_seq to 1
    opt.output_seq = 1
    class mydata(Dataset):
        def __init__(self, datalist, input_seq):
            self.data = datalist
            self.input_seq = input_seq
            # Setting a random seed to ensure that the lead-time will be the same across runs
            np.random.seed(10)
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            batch = self.data[idx]
            randint = np.random.randint(0,6)
            lead_time = torch.ones((1,self.input_seq+6,256,256))*randint
            final = torch.cat((batch,lead_time),0)
            input_ = final[:,0:self.input_seq,]
            target = final[0,self.input_seq+randint:self.input_seq+randint+1,64:192, 64:192]
            #Reshaping inpu-size by collapsing channels,input_seq to one dimension of channels*input_seq (concatanating)
            input_ = input_.reshape((input_.shape[0]*input_.shape[1],input_.shape[2],input_.shape[3]))
            return input_.type(datatype), target.type(datatype)
        
    class mydataTest(Dataset):
        def __init__(self, datalist, input_seq):
            self.data = datalist
            self.input_seq = input_seq
            # Setting a random seed to ensure that the lead-time will be the same across runs
            #np.random.seed(10)
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            batch = self.data[idx]
            randint = opt.leadtime
            lead_time = torch.ones((1,self.input_seq+6,256,256))*randint
            final = torch.cat((batch,lead_time),0)
            # del lead_time
            input_ = final[:,0:self.input_seq,]
            target = final[0,self.input_seq+randint:self.input_seq+randint+1,64:192, 64:192]
            #Reshaping inpu-size by collapsing channels,input_seq to one dimension of channels*input_seq (concatanating)
            input_ = input_.reshape((input_.shape[0]*input_.shape[1],input_.shape[2],input_.shape[3]))
            return input_.type(datatype), target.type(datatype)
    
#Shuffle data before splitting into training/validation/testing
np.random.shuffle(chunklist)   

   
# Use 80% of data for training    
train_cut = int((len(chunklist)*0.8//1))
# Use 10% for validation and 10% for testing    
val_cut = int((len(chunklist)*0.1//1))
# Slicing data
train_data= chunklist[:train_cut]
val_data= chunklist[train_cut:train_cut+val_cut]
test_data = chunklist[train_cut+val_cut:]

##############################################################################

#Overwrite setting for lr-finder if we continue from checkpoint
if opt.continue_from_latest:
    opt.auto_lr = False

print("\nWriting log-file")
path = opt.model_name+"/"+opt.model_description+"/"
if not os.path.isdir(path):
    os.makedirs(path)

from datetime import datetime
time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
run_time = "Run Time: " + time_stamp + "\n"

with open(path+"log-file.txt", "a") as f:
    f.writelines([run_time] + [arg+"="+str(getattr(opt, arg))+"\n" for arg in vars(opt)])

################################### Model ####################################

# Attention modules for CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# Basic Conv-BatchNorm-ReLU Block
class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Encoder Block
class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )
        self.ca = ChannelAttention(in_planes = y_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.encode(x)
        attended_x = self.ca(x) * x #automatically broadcasted
        attended_x = self.sa(attended_x) * attended_x #automatically broadcasted
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return attended_x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, last=False):
        super(StackDecoder, self).__init__()
        self.last = last
        padding = (kernel_size - 1) // 2

        self.up = nn.ConvTranspose2d(x_channels, x_channels, kernel_size=2, stride=2)

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        if not self.last:
            x = self.up(x)
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x


# UNet
class UNet(pl.LightningModule):
    def __init__(self, input_dim, kernel_size, input_seq, output_seq, lr, batch_size):
        super(UNet, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.batch_size = batch_size
        self.down1 = StackEncoder(input_dim*input_seq, 24, kernel_size=kernel_size)  # 128 (input = 256)
        self.down2 = StackEncoder(24, 64, kernel_size=kernel_size)  # 64
        self.down3 = StackEncoder(64, 128, kernel_size=kernel_size)  # 32
        self.down4 = StackEncoder(128, 256, kernel_size=kernel_size)  # 16
        self.down5 = StackEncoder(256, 512, kernel_size=kernel_size)  # 8

        self.up5 = StackDecoder(512, 512, 256, kernel_size=kernel_size)  # 16
        self.up4 = StackDecoder(256, 256, 128, kernel_size=kernel_size)  # 32
        self.up3 = StackDecoder(128, 128, 64, kernel_size=kernel_size)  # 64
        self.up2 = StackDecoder(64, 64, 24, kernel_size=kernel_size)  # 128
        self.up1 = StackDecoder(24, 24, 24, kernel_size=kernel_size, last = True)  # 128

        self.classify = nn.Conv2d(24, output_seq, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=kernel_size, padding=1))

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        transform1 = CenterCrop(128)
        down1 = transform1(down1)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)

        out = self.up5(out, down5)
        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        # Making the channels the output_seq by adding a dimension for the channel_dim which is 1 (cloud mask)
        out = torch.unsqueeze(out, 1)

        return out
        
    def create_grid(self, output, target):
        target = target[0].transpose(0,1)
        output = output[0].transpose(0,1)
        
        final = torch.cat([target,output],dim=0)
        grid = torchvision.utils.make_grid([final[i] for i in range(final.shape[0])], nrow=self.output_seq)
        
        return grid
    
    def save_grid(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = torchvision.transforms.functional.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        path = opt.model_name+"/"+opt.model_description+"/savedImages/"
        if not os.path.isdir(path):
            os.makedirs(path)
        fig.savefig(path+"Epoch_"+str(self.current_epoch)+".png")
        plt.close()
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, betas=(opt.beta_1, opt.beta_2))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "avg_val_loss"}

    def train_dataloader(self):
        train_loader = DataLoader(mydata(train_data, input_seq = self.input_seq), shuffle=True, 
                        batch_size=self.batch_size, num_workers = 8)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(mydata(val_data, input_seq = self.input_seq), shuffle=False, 
                        batch_size=self.batch_size, num_workers = 8)
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(mydataTest(test_data, input_seq = self.input_seq), shuffle=False, 
                        batch_size=self.batch_size, num_workers = 8)
        return test_loader
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_, target = batch
        target = torch.unsqueeze(target, 1)
        output = self(input_)
        loss1 = F.binary_cross_entropy_with_logits(output.flatten(), target.flatten())
        loss2 = F.l1_loss(torch.sigmoid(output).flatten(), target.flatten())
        loss = opt.lambda_BCE*loss1 + opt.lambda_L1 * loss2 
        # Logging to TensorBoard by default
        #self.log("train_loss", loss)
        #tensorboard_logs = {'train_BCE_loss': loss}

        #return {'loss': loss, 'log': tensorboard_logs}
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train_Loss", avg_loss, self.current_epoch)
        #self.logger.experiment.add_scalar("LearningRate", self.lr, self.current_epoch)
        #   return {"avg_loss": avg_loss}
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_, target = batch
        target = torch.unsqueeze(target, 1)
        output = self(input_)
        loss1 = F.binary_cross_entropy_with_logits(output.flatten(), target.flatten())
        loss2 = F.l1_loss(torch.sigmoid(output).flatten(), target.flatten())
        loss = opt.lambda_BCE*loss1 + opt.lambda_L1 * loss2
        if self.current_epoch % 5 == 0:
            grid = self.create_grid(torch.sigmoid(output), target)
            self.save_grid(grid)
        # Logging to TensorBoard by default
        #self.log("val_loss", loss)
        #tensorboard_logs = {'val_BCE_loss': loss}
        #return {"loss": loss, "log": tensorboard_logs}
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation_Loss", avg_loss, self.current_epoch)
        #used to monitor lr_scheduler
        self.log("avg_val_loss", avg_loss)
        return {'avg_val_loss': avg_loss}
        
    ############################SSIM FUNCTIONS##################################################################
    
    def gaussian(self,window_size, sigma):
        """
        Generates a list of Tensor values drawn from a gaussian distribution with standard
        diviation = sigma and sum of all elements = 1.

        Length of list = window_size
        """    
        gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
        
    def create_window(self,window_size, channel=1):

        # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = self.gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
        
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
         
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

        return window
    
    def ssim(self,img1, img2, window_size=11, window=None, size_average=True, full=False):

        L = 1 # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

        pad = window_size // 2
        
        try:
            _, channels, height, width = img1.size()
        except:
            channels, height, width = img1.size()

        # if window is not provided, init one
        if window is None: 
            real_size = min(window_size, height, width) # window should be atleast 11x11 
            window = self.create_window(real_size, channel=channels).to(img1.device)
        
        # calculating the mu parameter (locally) for both images using a gaussian filter 
        # calculates the luminosity params
        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2 
        mu12 = mu1 * mu2

        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component 
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

        # Some constants for stability 
        C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
        C2 = (0.03 ) ** 2 

        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)

        numerator1 = 2 * mu12 + C1  
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1 
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

        if size_average:
            ret = ssim_score.mean() 
        else: 
            ret = ssim_score.mean(1).mean(1).mean(1)
        
        if full:
            return ret, contrast_metric
        
        return ret
        
    ############################################################################################################
        
    def test_step(self, batch, batch_idx):
        input_, target = batch
        target = torch.unsqueeze(target, 1)
        output = self(input_)
        loss = F.binary_cross_entropy_with_logits(output.flatten(), target.flatten())
        mse_loss = F.mse_loss(torch.sigmoid(output).flatten(), target.flatten(), reduction = "sum")
        if opt.version == 1:
            #MSE_Loss pr. frame
            mse_loss1 = F.mse_loss(torch.sigmoid(output[:,:,0,:,:]).flatten(), target[:,:,0,:,:].flatten(), reduction = "sum")
            mse_loss2 = F.mse_loss(torch.sigmoid(output[:,:,1,:,:]).flatten(), target[:,:,1,:,:].flatten(), reduction = "sum")
            mse_loss3 = F.mse_loss(torch.sigmoid(output[:,:,2,:,:]).flatten(), target[:,:,2,:,:].flatten(), reduction = "sum")
            mse_loss4 = F.mse_loss(torch.sigmoid(output[:,:,3,:,:]).flatten(), target[:,:,3,:,:].flatten(), reduction = "sum")
            mse_loss5 = F.mse_loss(torch.sigmoid(output[:,:,4,:,:]).flatten(), target[:,:,4,:,:].flatten(), reduction = "sum")
            mse_loss6 = F.mse_loss(torch.sigmoid(output[:,:,5,:,:]).flatten(), target[:,:,5,:,:].flatten(), reduction = "sum")
        #Average PSNR for the frames in the batch
        PSNR = sum([20*math.log10(1/math.sqrt(F.mse_loss(torch.sigmoid(output[i,:,j]).flatten(),target[i,:,j].flatten()).detach().item())) for i in range(target.shape[0]) for j in range(target.shape[2])])
        SSIM = sum([self.ssim(torch.sigmoid(output[i,:,j]).unsqueeze(0), target[i,:,j].unsqueeze(0)).detach().item() for i in range(target.shape[0]) for j in range(target.shape[2])])
        if opt.version == 1:
            #SSIM pr frame
            SSIM1 = sum([self.ssim(torch.sigmoid(output[i,:,0]).unsqueeze(0), target[i,:,0].unsqueeze(0)).detach().item() for i in range(target.shape[0])])
            SSIM2 = sum([self.ssim(torch.sigmoid(output[i,:,1]).unsqueeze(0), target[i,:,1].unsqueeze(0)).detach().item() for i in range(target.shape[0])])
            SSIM3 = sum([self.ssim(torch.sigmoid(output[i,:,2]).unsqueeze(0), target[i,:,2].unsqueeze(0)).detach().item() for i in range(target.shape[0])])
            SSIM4 = sum([self.ssim(torch.sigmoid(output[i,:,3]).unsqueeze(0), target[i,:,3].unsqueeze(0)).detach().item() for i in range(target.shape[0])])
            SSIM5 = sum([self.ssim(torch.sigmoid(output[i,:,4]).unsqueeze(0), target[i,:,4].unsqueeze(0)).detach().item() for i in range(target.shape[0])])
            SSIM6 = sum([self.ssim(torch.sigmoid(output[i,:,5]).unsqueeze(0), target[i,:,5].unsqueeze(0)).detach().item() for i in range(target.shape[0])])
        
        #Calculating mean acc.
        output2 = torch.sigmoid(output)
        output2[output2 < 0.5] = 0
        output2[output2 >= 0.5] = 1
        
        acc = torch.sum(output2.flatten() == target.flatten())
        
        # Logging to TensorBoard by default
        #self.log("val_loss", loss)
        #tensorboard_logs = {'val_BCE_loss': loss}
        #return {"loss": loss, "log": tensorboard_logs}
        return {"MSE_test_loss": mse_loss, 
                "BCE_test_loss": loss, 
                "data_points": input_.shape[0], 
                "PSNR": PSNR, 
                "SSIM" : SSIM, 
                "Accuracy": acc,
                "MSE_test_loss1": mse_loss1 if opt.version == 1 else 0,
                "MSE_test_loss2": mse_loss2 if opt.version == 1 else 0,
                "MSE_test_loss3": mse_loss3 if opt.version == 1 else 0,
                "MSE_test_loss4": mse_loss4 if opt.version == 1 else 0,
                "MSE_test_loss5": mse_loss5 if opt.version == 1 else 0,
                "MSE_test_loss6": mse_loss6 if opt.version == 1 else 0,
                "SSIM1" : SSIM1 if opt.version == 1 else 0,
                "SSIM2" : SSIM2 if opt.version == 1 else 0,
                "SSIM3" : SSIM3 if opt.version == 1 else 0,
                "SSIM4" : SSIM4 if opt.version == 1 else 0,
                "SSIM5" : SSIM5 if opt.version == 1 else 0,
                "SSIM6" : SSIM6 if opt.version == 1 else 0}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["BCE_test_loss"] for x in outputs]).mean()
        total_mse_loss = torch.stack([x["MSE_test_loss"] for x in outputs]).sum()
        
        total_datapoints = sum([x["data_points"] for x in outputs])
        print("Total datapoints is:", total_datapoints)
        if opt.version == 1:
            #Pr. frame mse
            total_mse_loss1 = torch.stack([x["MSE_test_loss1"] for x in outputs]).sum()
            total_mse_loss2 = torch.stack([x["MSE_test_loss2"] for x in outputs]).sum()
            total_mse_loss3 = torch.stack([x["MSE_test_loss3"] for x in outputs]).sum()
            total_mse_loss4 = torch.stack([x["MSE_test_loss4"] for x in outputs]).sum()
            total_mse_loss5 = torch.stack([x["MSE_test_loss5"] for x in outputs]).sum()
            total_mse_loss6 = torch.stack([x["MSE_test_loss6"] for x in outputs]).sum()
            accuracy = sum([x["Accuracy"] for x in outputs])/(6*total_datapoints*128*128)
            mse_pr_frame = total_mse_loss / (total_datapoints*6*128*128)
            avgPSNR = sum([x["PSNR"] for x in outputs])/(6*total_datapoints)
            avgSSIM = sum([x["SSIM"] for x in outputs])/(6*total_datapoints)
            #Per frame SSIM
            avgSSIM1 = sum([x["SSIM1"] for x in outputs])/(total_datapoints)
            avgSSIM2 = sum([x["SSIM2"] for x in outputs])/(total_datapoints)
            avgSSIM3 = sum([x["SSIM3"] for x in outputs])/(total_datapoints)
            avgSSIM4 = sum([x["SSIM4"] for x in outputs])/(total_datapoints)
            avgSSIM5 = sum([x["SSIM5"] for x in outputs])/(total_datapoints)
            avgSSIM6 = sum([x["SSIM6"] for x in outputs])/(total_datapoints)
            print("SSIM for frame 1:", avgSSIM1)
            print("SSIM for frame 2:", avgSSIM2)
            print("SSIM for frame 3:", avgSSIM3)
            print("SSIM for frame 4:", avgSSIM4)
            print("SSIM for frame 5:", avgSSIM5)
            print("SSIM for frame 6:", avgSSIM6)
            #Per frame MSE
            mse_pr_frame1 = total_mse_loss1 / (total_datapoints*128*128)
            mse_pr_frame2 = total_mse_loss2 / (total_datapoints*128*128)
            mse_pr_frame3 = total_mse_loss3 / (total_datapoints*128*128)
            mse_pr_frame4 = total_mse_loss4 / (total_datapoints*128*128)
            mse_pr_frame5 = total_mse_loss5 / (total_datapoints*128*128)
            mse_pr_frame6 = total_mse_loss6 / (total_datapoints*128*128)
            print("MSE for frame 1:", mse_pr_frame1)
            print("MSE for frame 2:", mse_pr_frame2)
            print("MSE for frame 3:", mse_pr_frame3)
            print("MSE for frame 4:", mse_pr_frame4)
            print("MSE for frame 5:", mse_pr_frame5)
            print("MSE for frame 6:", mse_pr_frame6)
            
        else:
            accuracy = sum([x["Accuracy"] for x in outputs])/(total_datapoints*128*128)
            mse_pr_frame = total_mse_loss / (total_datapoints*128*128)
            avgPSNR = sum([x["PSNR"] for x in outputs])/total_datapoints
            avgSSIM = sum([x["SSIM"] for x in outputs])/total_datapoints
            print("With lead-time:", opt.leadtime)
        print("Average MSE pr. frame is:", mse_pr_frame)
        print("Average BCE Loss is:", avg_loss)
        print("Average PSNR is:", avgPSNR)
        print("Average SSIM pr. frame is:", avgSSIM)
        print("Accuracy is ", accuracy)
        return {"avg_test_loss":avg_loss, "MSE_pr_frame":  mse_pr_frame}
        
    
##############################################################################

    
#####################################END OF MODEL #########################################

def run_trainer():
    if not opt.testing:
        logger = TensorBoardLogger(opt.model_name, opt.model_description)
        epoch_callback = ModelCheckpoint(filename = opt.model_name+"-{epoch}", dirpath=opt.model_name+"/"+opt.model_description+"/savedCheckpoints/", every_n_epochs = 1, save_top_k=-1)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        if opt.continue_from_latest:
            checkpoints = os.listdir(opt.model_name+"/"+opt.model_description+"/savedCheckpoints")
            sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split("=")[1].split(".")[0]))
            latest_checkpoint = sorted_checkpoints[-1]
            c_path = opt.model_name+"/"+opt.model_description+"/savedCheckpoints/"+latest_checkpoint
            trainer = Trainer(fast_dev_run = False, gpus=opt.n_gpus, max_epochs = opt.epochs, precision=16, logger=logger, callbacks = [epoch_callback,lr_monitor], resume_from_checkpoint=c_path, num_sanity_val_steps=0, accumulate_grad_batches=opt.gradbatches)
            model = UNet.load_from_checkpoint(c_path)
        else:
            trainer = Trainer(fast_dev_run = False, gpus=opt.n_gpus, max_epochs = opt.epochs, precision=16, logger=logger, callbacks = [epoch_callback,lr_monitor], auto_lr_find = opt.auto_lr, num_sanity_val_steps=0, accumulate_grad_batches=opt.gradbatches)
            model = UNet(opt.input_dim, opt.kernel_size, opt.input_seq, opt.output_seq, opt.lr, opt.batch_size)
            
        if opt.auto_lr:
            trainer.tune(model)
        trainer.fit(model)
    else:
        checkpoints = os.listdir(opt.model_name+"/"+opt.model_description+"/savedCheckpoints")
        sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split("=")[1].split(".")[0]))
        latest_checkpoint = sorted_checkpoints[-1]
        c_path = opt.model_name+"/"+opt.model_description+"/savedCheckpoints/"+latest_checkpoint
        
        model_test = UNet.load_from_checkpoint(c_path, input_dim=opt.input_dim, kernel_size=opt.kernel_size, input_seq=opt.input_seq, output_seq=opt.output_seq, lr=opt.lr, batch_size=opt.batch_size)
        
        trainer = Trainer(gpus = opt.n_gpus, precision=16)
        trainer.test(model_test)


if __name__ == '__main__':
    run_trainer()
        