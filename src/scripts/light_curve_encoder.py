"""
Training script for the light curve encoder.
The network takes in light curves images and build encoded
representation, which will be further utilize in AGNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
import datetime
import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

folder = "/media/joshua/Milano/DR7_QSO_LCs_delta_t/"  # path to the dataset 
EPOCH = 100
glo_batch_size = 10
test_num_batch = 50

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            normalize,
            ])
target_transform = torch.Tensor

class BHDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_folder = 'train'#'data_train'
        self.test_folder = 'val'#'data_test'

        #self.df = pd.read_csv(self.root_dir + '/clean_full_data.csv')

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.df = pd.read_csv("/media/joshua/Milano/DR7_features_DRW/train" + '/train.csv') # path to csv table for the training set
            #self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv("/media/joshua/Milano/DR7_features_DRW/val" + '/val.csv') # path to csv table for the training set
            #self.length = TESTING_SAMPLES

    def __getitem__(self, index):

        #print(self.df['ID'])
        ID = self.df['ID'].iloc[[index]]
        M = self.df['Mass_ground_truth'].iloc[[index]]
        M_ERR = self.df['M_ERR'].iloc[[index]]
        z = self.df['z'].iloc[[index]]
        M_i = self.df['M_i'].iloc[[index]]
        u_band = self.df['u_band'].iloc[[index]]
        g_band = self.df['g_band'].iloc[[index]]
        r_band = self.df['r_band'].iloc[[index]]
        i_band = self.df['i_band'].iloc[[index]]
        z_band = self.df['z_band'].iloc[[index]]
        u_band_std = self.df['u_band_std'].iloc[[index]]
        g_band_std = self.df['g_band_std'].iloc[[index]]
        r_band_std = self.df['r_band_std'].iloc[[index]]
        i_band_std = self.df['i_band_std'].iloc[[index]]
        z_band_std = self.df['z_band_std'].iloc[[index]]
        u_g = self.df['u-g'].iloc[[index]]
        g_r = self.df['g-r'].iloc[[index]]
        r_i = self.df['r-i'].iloc[[index]]
        i_z = self.df['i-z'].iloc[[index]]
        z_u = self.df['z-u'].iloc[[index]]
        tau = self.df['tau'].iloc[[index]]
        sigma = self.df['sigma'].iloc[[index]]

        img_path = "/media/joshua/Milano/Efficient_Full_train" + '/LC_images_' + str(ID.values[0]) + '.npy' ## path to the light curve images
        img = np.load(img_path)
        image = np.zeros((3, 224, 224))
        for i in range(3):
            image[i, :, :] += img


        return image, M.values, z.values

    def __len__(self):
        return self.df.shape[0]



train_loader = torch.utils.data.DataLoader(BHDataset(folder, train=True, transform=data_transform, target_transform=target_transform),
                    batch_size = glo_batch_size, shuffle = True
                    )

if __name__ == '__main__':

    dset_classes_number = 1
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Sequential(
    nn.Linear(num_ftrs, 10),
    nn.Linear(10, dset_classes_number))
    loss_fn = nn.MSELoss(reduction='mean')

    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr = 1e-3)
    tb = SummaryWriter()

    best_accuracy = float("inf")


    #if os.path.exists('./saved_model/resnet18.mdl'):
        #net = torch.load('./saved_model/resnet18.mdl')
        #print('loaded mdl!')

    for epoch in range(EPOCH):

        net.train()
        total_loss = 0.0
        total_counter = 0
        total_rms = 0

        for batch_idx, (data, BH_Mass, z) in enumerate(tqdm(train_loader, total = len(train_loader))):
            data, target = data.float(), z.float()
            data, target = Variable(data).cuda(), Variable(target).cuda()

            optimizer.zero_grad()
            output = net(data)
            loss = loss_fn(output, target)

            square_diff = (output - target) #((output - target)**2)**(0.5)
            total_rms += square_diff.std(dim=0)
            total_loss += loss.item()
            total_counter += 1

            loss.backward()
            optimizer.step()

        # Collect RMS over each label
        avg_rms = total_rms / (total_counter)
        avg_rms = avg_rms.cpu()
        avg_rms = (avg_rms.data).numpy()
        for i in range(len(avg_rms)):
            tb.add_scalar('rms %d' % (i+1), avg_rms[i])

        # print test loss and tets rms
        print(epoch, 'Train loss (averge per batch wise):', total_loss/(total_counter), ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))

        with torch.no_grad():
            net.eval()
            total_loss = 0.0
            total_counter = 0
            total_rms = 0

            test_loader = torch.utils.data.DataLoader(BHDataset(folder, train=False, transform=data_transform, target_transform=target_transform),
                        batch_size = glo_batch_size, shuffle = True
                        )

            for batch_idx, (data, BH_Mass, z) in enumerate(test_loader):
                data, target = data.float(), z.float()
                data, target = Variable(data).cuda(), Variable(target).cuda()

                #pred [batch, out_caps_num, out_caps_size, 1]
                pred = net(data)
                loss = loss_fn(pred, target)
                square_diff = (pred - target)
                total_rms += square_diff.std(dim=0)
                total_loss += loss.item()
                total_counter += 1

                if batch_idx % test_num_batch == 0 and batch_idx != 0:
                    tb.add_scalar('test_loss', loss.item())
                    break

            # Collect RMS over each label
            avg_rms = total_rms / (total_counter)
            avg_rms = avg_rms.cpu()
            avg_rms = (avg_rms.data).numpy()
            for i in range(len(avg_rms)):
                tb.add_scalar('rms %d' % (i+1), avg_rms[i])

            # print test loss and tets rms
            print(epoch, 'Test loss (averge per batch wise):', total_loss/(total_counter), ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))
            if total_loss/(total_counter) < best_accuracy:
                best_accuracy = total_loss/(total_counter)
                datetime_today = str(datetime.date.today())
                torch.save(net, './saved_model/' + datetime_today + 'z_lc_image_resnet18_encoder.mdl')
                print("saved to " + "z_lc_image_resnet18_encoder.mdl" + " file.")

    tb.close()
