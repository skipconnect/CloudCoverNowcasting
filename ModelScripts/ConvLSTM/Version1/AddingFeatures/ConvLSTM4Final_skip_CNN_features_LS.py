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

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--factor', type=float, default=0.1, help='plateau factor')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')
parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for ConvLSTM layers')
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
parser.add_argument('--patience', type=int, default=3, help='Patience in LR-scheduler')
parser.add_argument('--save_batch', type=int, default=7, help='Batch_idx to save results from')
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
    if opt.version == 1:
        opt.input_dim = 2
    else:
        opt.input_dim = 3 

##############################################################################
# Defining datasets

#Setting a random.seed (this is for the shuffeling of the dataset, to ensure it is the same across each run)
np.random.seed(12)
datatype = torch.float16
if opt.version == 1:
    class mydata(Dataset):
        def __init__(self, datalist, input_seq):
            self.data= datalist
            self.input_seq = input_seq
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            batch = self.data[idx]
            #input_ = batch[:2,:self.input_seq,]
            cloud = batch[0,:self.input_seq,].unsqueeze(0)
            ls = batch[2,:self.input_seq,].unsqueeze(0)
            input_ = torch.cat([cloud,ls])
            target = batch[0,self.input_seq:,64:192, 64:192]
            return input_.type(datatype), target.type(datatype)
            
    class mydataTest(Dataset):
        def __init__(self, datalist, input_seq):
            self.data= datalist
            self.input_seq = input_seq
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            batch = self.data[idx]
            #input_ = batch[:,:self.input_seq,]
            cloud = batch[0,:self.input_seq,].unsqueeze(0)
            ls = batch[2,:self.input_seq,].unsqueeze(0)
            input_ = torch.cat([cloud,ls])
            target = batch[0,self.input_seq:,64:192, 64:192]
            return input_.type(datatype), target.type(datatype)


if opt.version == 2:
    # Since we are only predicting one frame we set output_seq to 1
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
            input_ = final[:2,0:self.input_seq,]
            target = final[0,self.input_seq+randint:self.input_seq+randint+1,64:192, 64:192]
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

#Creating conv2d with weight standardization
class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(pl.LightningModule):
    """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

    def __init__(self, input_dim, hidden_dim, kernel_size, frame_size):
        super(ConvLSTMCell, self).__init__()  
        
        self.hidden_dim = hidden_dim
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = Conv2d(
            in_channels=input_dim + hidden_dim, 
            out_channels=4 * hidden_dim, 
            kernel_size=kernel_size, 
            padding="same")
        self.gn = nn.GroupNorm(4 * hidden_dim // 32, 4 * hidden_dim )

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.zeros(hidden_dim, frame_size,frame_size))
        self.W_co = nn.Parameter(torch.zeros(hidden_dim, frame_size,frame_size))
        self.W_cf = nn.Parameter(torch.zeros(hidden_dim, frame_size,frame_size))

    def forward(self, X, H_prev, C_prev):
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        conv_output = self.gn(conv_output)
        
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
        
        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * torch.tanh(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * torch.tanh(C)
        return H, C
        
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# Defining the spatial downsampler
class spatialdownsampling(nn.Module):
    def __init__(self, input_dim):
        super(spatialdownsampling, self).__init__()
        self.input_dim = input_dim
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(self.input_dim, 32, kernel_size=(1,3,3), padding=(0,1,1)) #256
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(1,3,3), padding=(0,1,1))#256
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(1,3,3), padding=(0,1,1))#256
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64,64, kernel_size=(1,2,2), stride= (1,2,2), padding=0) #128
        self.bn4 = nn.BatchNorm3d(64)
        self.conv5 = nn.Conv3d(64,64, kernel_size=(1,3,3), padding=(0,1,1)) #128
        self.bn5 = nn.BatchNorm3d(64)
        self.conv6 = nn.Conv3d(64, 1, kernel_size=(1,3,3), padding=(0,1,1)) #128
        self.bn6 = nn.BatchNorm3d(1)
        self.transformC = CenterCrop(128)

    def forward(self, x):
        x_skip = self.transformC(x)
        #x = x[:,0,].unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        #x = self.bn6(x)
        #x = self.relu(x)
        x_ret = torch.cat([x_skip,x], 1)
        return x_ret

# Defining the feature extractor
class featureextractor(nn.Module):
    def __init__(self, input_dim):
        super(featureextractor, self).__init__()
        self.input_dim = input_dim
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(self.input_dim+1, 32, kernel_size=(1,3,3), padding=(0,1,1)) #128
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(1,3,3), padding=(0,1,1))#128
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(1,3,3), padding=(0,1,1))#128
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64,64, kernel_size=(1,3,3), padding=(0,1,1)) #128
        self.bn4 = nn.BatchNorm3d(64)
        self.conv5 = nn.Conv3d(64,64, kernel_size=(1,3,3), padding=(0,1,1)) #128
        self.bn5 = nn.BatchNorm3d(64)
        self.conv6 = nn.Conv3d(64, self.input_dim, kernel_size=(1,3,3), padding=(0,1,1)) #128
        self.bn6 = nn.BatchNorm3d(self.input_dim)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        #x = self.bn6(x)
        #x = self.relu(x)
        return x

    
# Two-layer ConvLSTM encoder-forecaster model (Model 4)
class EncoderDecoderConvLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, kernel_size, frame_size, input_seq, output_seq, lr, batch_size):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.lr = lr
        self.batch_size = batch_size
        self.save_hyperparameters()
        
        self.input_seq = input_seq
        self.output_seq = output_seq
        
        self.spatialdown = spatialdownsampling(input_dim)
        self.feature = featureextractor(input_dim)
      
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=input_dim,
                                               hidden_dim=hidden_dim,
                                               kernel_size=kernel_size,
                                               frame_size = frame_size)

        

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=hidden_dim,
                                               hidden_dim=hidden_dim,
                                               kernel_size=kernel_size,
                                               frame_size = frame_size)



        self.decoder_1_convlstm = ConvLSTMCell(input_dim=hidden_dim,
                                               hidden_dim=hidden_dim,
                                               kernel_size=kernel_size,
                                               frame_size = frame_size)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=hidden_dim,
                                               hidden_dim=hidden_dim,
                                               kernel_size=kernel_size,
                                               frame_size = frame_size)
        

        self.decoder_CNN = nn.Conv3d(in_channels=hidden_dim,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []
        # encoder
        for t in range(self.input_seq):
            h_t, c_t = self.encoder_1_convlstm(X=x[:, :, t, :],
                                               H_prev=h_t, C_prev = c_t)  # we could concat to provide skip conn here
            
            h_t2, c_t2 = self.encoder_2_convlstm(X=h_t,
                                                 H_prev=h_t2, C_prev = c_t2)  # we could concat to provide skip conn here
            
            
          
        # encoder_vector
        encoder_vector = h_t2

        # Forecaster
        input_next = encoder_vector 
        for t in range(self.output_seq):
            h_t3, c_t3 = self.decoder_1_convlstm(X=input_next,
                                                 H_prev=h_t3, C_prev = c_t3)  # we could concat to provide skip conn here
            
            h_t4, c_t4 = self.decoder_2_convlstm(X=h_t3,
                                                 H_prev=h_t4, C_prev = c_t4)  # we could concat to provide skip conn here
            
            input_next = h_t4
            outputs += [h_t4]  # predictions
        
        outputs = torch.stack(outputs, 2)

        #outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        #outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
       
        x = self.spatialdown(x)
        
        x = self.feature(x)
        
        # find size of different input dimensions
        b, _, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, h_t, c_t,h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs
    
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = opt.patience, factor = opt.factor)
        
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
        loss = F.binary_cross_entropy_with_logits(output.flatten(), target.flatten())
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
        loss = F.binary_cross_entropy_with_logits(output.flatten(), target.flatten())
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
        #Saving test results
        if batch_idx == opt.save_batch:
            torch.save(input_.cpu(),opt.model_name+"/"+opt.model_description+"/savedTensors/Input_batch"+str(opt.save_batch)+".pt")
            torch.save(target.cpu(),opt.model_name+"/"+opt.model_description+"/savedTensors/Target_batch"+str(opt.save_batch)+".pt")
            torch.save(torch.sigmoid(output).cpu(),opt.model_name+"/"+opt.model_description+"/savedTensors/Output_batch"+str(opt.save_batch)+".pt")
        
        if opt.version == 1:
            #MSE_Loss pr. frame
            mse_loss1 = F.mse_loss(torch.sigmoid(output[:,:,0,:,:]).flatten(), target[:,:,0,:,:].flatten(), reduction = "sum")
            mse_loss1_full = F.mse_loss(torch.sigmoid(output[:,:,0,:,:]), target[:,:,0,:,:], reduction = "none")
            mse_loss2 = F.mse_loss(torch.sigmoid(output[:,:,1,:,:]).flatten(), target[:,:,1,:,:].flatten(), reduction = "sum")
            mse_loss2_full = F.mse_loss(torch.sigmoid(output[:,:,1,:,:]), target[:,:,1,:,:], reduction = "none")
            mse_loss3 = F.mse_loss(torch.sigmoid(output[:,:,2,:,:]).flatten(), target[:,:,2,:,:].flatten(), reduction = "sum")
            mse_loss3_full = F.mse_loss(torch.sigmoid(output[:,:,2,:,:]), target[:,:,2,:,:], reduction = "none")
            mse_loss4 = F.mse_loss(torch.sigmoid(output[:,:,3,:,:]).flatten(), target[:,:,3,:,:].flatten(), reduction = "sum")
            mse_loss4_full = F.mse_loss(torch.sigmoid(output[:,:,3,:,:]), target[:,:,3,:,:], reduction = "none")
            mse_loss5 = F.mse_loss(torch.sigmoid(output[:,:,4,:,:]).flatten(), target[:,:,4,:,:].flatten(), reduction = "sum")
            mse_loss5_full = F.mse_loss(torch.sigmoid(output[:,:,4,:,:]), target[:,:,4,:,:], reduction = "none")
            mse_loss6 = F.mse_loss(torch.sigmoid(output[:,:,5,:,:]).flatten(), target[:,:,5,:,:].flatten(), reduction = "sum")
            mse_loss6_full = F.mse_loss(torch.sigmoid(output[:,:,5,:,:]), target[:,:,5,:,:], reduction = "none")
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
                "MSE_test_loss1_full": mse_loss1_full if opt.version == 1 else 0,
                "MSE_test_loss2": mse_loss2 if opt.version == 1 else 0,
                "MSE_test_loss2_full": mse_loss2_full if opt.version == 1 else 0,
                "MSE_test_loss3": mse_loss3 if opt.version == 1 else 0,
                "MSE_test_loss3_full": mse_loss3_full if opt.version == 1 else 0,
                "MSE_test_loss4": mse_loss4 if opt.version == 1 else 0,
                "MSE_test_loss4_full": mse_loss4_full if opt.version == 1 else 0,
                "MSE_test_loss5": mse_loss5 if opt.version == 1 else 0,
                "MSE_test_loss5_full": mse_loss5_full if opt.version == 1 else 0,
                "MSE_test_loss6": mse_loss6 if opt.version == 1 else 0,
                "MSE_test_loss6_full": mse_loss6_full if opt.version == 1 else 0,
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
            total_mse_loss1_full = torch.cat([x["MSE_test_loss1_full"] for x in outputs],0).sum(dim = 0)
            torch.save((total_mse_loss1_full / total_datapoints),opt.model_name+"/"+opt.model_description+"/savedMseTensors/1.pt")
            total_mse_loss2 = torch.stack([x["MSE_test_loss2"] for x in outputs]).sum()
            total_mse_loss2_full = torch.cat([x["MSE_test_loss2_full"] for x in outputs],0).sum(dim = 0)
            torch.save((total_mse_loss2_full / total_datapoints),opt.model_name+"/"+opt.model_description+"/savedMseTensors/2.pt")
            total_mse_loss3 = torch.stack([x["MSE_test_loss3"] for x in outputs]).sum()
            total_mse_loss3_full = torch.cat([x["MSE_test_loss3_full"] for x in outputs],0).sum(dim = 0)
            torch.save((total_mse_loss3_full / total_datapoints),opt.model_name+"/"+opt.model_description+"/savedMseTensors/3.pt")
            total_mse_loss4 = torch.stack([x["MSE_test_loss4"] for x in outputs]).sum()
            total_mse_loss4_full = torch.cat([x["MSE_test_loss4_full"] for x in outputs],0).sum(dim = 0)
            torch.save((total_mse_loss4_full / total_datapoints),opt.model_name+"/"+opt.model_description+"/savedMseTensors/4.pt")
            total_mse_loss5 = torch.stack([x["MSE_test_loss5"] for x in outputs]).sum()
            total_mse_loss5_full = torch.cat([x["MSE_test_loss5_full"] for x in outputs],0).sum(dim = 0)
            torch.save((total_mse_loss5_full / total_datapoints),opt.model_name+"/"+opt.model_description+"/savedMseTensors/5.pt")
            total_mse_loss6 = torch.stack([x["MSE_test_loss6"] for x in outputs]).sum()
            total_mse_loss6_full = torch.cat([x["MSE_test_loss6_full"] for x in outputs],0).sum(dim = 0)
            torch.save((total_mse_loss6_full / total_datapoints),opt.model_name+"/"+opt.model_description+"/savedMseTensors/6.pt")
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
            model = EncoderDecoderConvLSTM.load_from_checkpoint(c_path)
        else:
            trainer = Trainer(fast_dev_run = False, gpus=opt.n_gpus, max_epochs = opt.epochs, precision=16, logger=logger, callbacks = [epoch_callback,lr_monitor], auto_lr_find = opt.auto_lr, num_sanity_val_steps=0, accumulate_grad_batches=opt.gradbatches)
            model = EncoderDecoderConvLSTM(opt.input_dim, opt.n_hidden_dim, opt.kernel_size, opt.frame_size, opt.input_seq, opt.output_seq, opt.lr, opt.batch_size)
            
        if opt.auto_lr:
            trainer.tune(model)
        trainer.fit(model)
    else:
        checkpoints = os.listdir(opt.model_name+"/"+opt.model_description+"/savedCheckpoints")
        checkpoints = [x for x in checkpoints if not x.startswith(".ipynb_")]
        sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split("=")[1].split(".")[0]))
        latest_checkpoint = sorted_checkpoints[-1]
        c_path = opt.model_name+"/"+opt.model_description+"/savedCheckpoints/"+latest_checkpoint
        model_test = EncoderDecoderConvLSTM.load_from_checkpoint(c_path)
        
        trainer = Trainer(gpus = opt.n_gpus, precision=16)
        trainer.test(model_test)


if __name__ == '__main__':
    run_trainer()
        