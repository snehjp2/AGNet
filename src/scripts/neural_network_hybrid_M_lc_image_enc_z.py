"""
Training script for the AGNet - a pytorch implementation of
hybrid combination with conv nets and multi-layer preception.
The network takes in features and light curves image and output AGN mass estimation, and save a mdl file.
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

folder = "/media/joshua/Milano/DR7_features_DRW/"
image_path = "/media/joshua/Milano/Efficient_Full_train/"
encoder_net_path = './saved_model/2021-04-21z_lc_image_resnet18_encoder.mdl'
EPOCH = 500
glo_batch_size = 10
test_num_batch = 50

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            normalize,
            ])
target_transform = torch.Tensor


class MLP(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, z_dim):
        super(MLP, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        self.fc4 = nn.Linear(h_dim3, h_dim4)
        self.fc5 = nn.Linear(h_dim4, h_dim5)
        self.fc51 = nn.Linear(h_dim5, z_dim)
        self.lrelu = nn.LeakyReLU(0.01, True)

    def encoder(self, x):

        h = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.lrelu(self.fc4(h))
        h = self.lrelu(self.fc5(h))

        return self.fc51(h)#, self.fc32(h) # mu, log_var

    def forward(self, x):
        output = self.encoder(x)
        return(output)


class BHDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, image_path, encoder_net_path, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.image_path = image_path
        self.encoder_net_path = encoder_net_path
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_folder = 'train'#'data_train'
        self.test_folder = 'val'#'data_test'
        self.encoder_net = torch.load(encoder_net_path)
        #self.df = pd.read_csv(self.root_dir + '/clean_full_data.csv')

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.df = pd.read_csv(self.path + '/train.csv')
            #self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + '/val.csv')
            #self.length = TESTING_SAMPLES
        ### for renormalize
        self.df_u_band_m = self.df['u_band'].mean()
        self.df_g_band_m = self.df['g_band'].mean()
        self.df_r_band_m = self.df['r_band'].mean()
        self.df_i_band_m = self.df['i_band'].mean()
        self.df_z_band_m = self.df['z_band'].mean()
        self.df_u_band_s = self.df['u_band'].std()
        self.df_g_band_s = self.df['g_band'].std()
        self.df_r_band_s = self.df['r_band'].std()
        self.df_i_band_s = self.df['i_band'].std()
        self.df_z_band_s = self.df['z_band'].std()
        self.df_u_band_std_m = self.df['u_band_std'].mean()
        self.df_g_band_std_m = self.df['g_band_std'].mean()
        self.df_r_band_std_m = self.df['r_band_std'].mean()
        self.df_i_band_std_m = self.df['i_band_std'].mean()
        self.df_z_band_std_m = self.df['z_band_std'].mean()
        self.df_u_band_std_s = self.df['u_band_std'].std()
        self.df_g_band_std_s = self.df['g_band_std'].std()
        self.df_r_band_std_s = self.df['r_band_std'].std()
        self.df_i_band_std_s = self.df['i_band_std'].std()
        self.df_z_band_std_s = self.df['z_band_std'].std()
        self.u_g_m = self.df['u-g'].mean()
        self.g_r_m = self.df['g-r'].mean()
        self.r_i_m = self.df['r-i'].mean()
        self.i_z_m = self.df['i-z'].mean()
        self.z_u_m = self.df['z-u'].mean()
        self.u_g_s = self.df['u-g'].std()
        self.g_r_s = self.df['g-r'].std()
        self.r_i_s = self.df['r-i'].std()
        self.i_z_s = self.df['i-z'].std()
        self.z_u_s = self.df['z-u'].std()
        self.z_m = self.df['z'].mean()
        self.M_i_m = self.df['M_i'].mean()
        self.tau_m = self.df['tau'].mean()
        self.sigma_m = self.df['sigma'].mean()
        self.z_s = self.df['z'].std()
        self.M_i_s = self.df['M_i'].std()
        self.tau_s = self.df['tau'].std()
        self.sigma_s = self.df['sigma'].std()

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


        ################################
        img_path = self.image_path + 'LC_images_' + str(ID.values[0]) + '.npy'
        img = np.load(img_path)
        image = np.zeros((3, 224, 224))
        for i in range(3):
            image[i, :, :] += img

        image = torch.from_numpy(image).float().cuda().unsqueeze(0)

        net = self.encoder_net

        x = net.conv1(image)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        layer1_output = net.layer1(x)
        layer2_output = net.layer2(layer1_output)
        layer3_output = net.layer3(layer2_output)
        layer4_output = net.layer4(layer3_output)
        x = net.avgpool(layer4_output)
        x = torch.flatten(x, 1)


        hidden_representation = net.fc[0](x)

        ################################

        features = np.zeros(29)
        features[0] = (u_band.values - self.df_u_band_m) / self.df_u_band_s
        features[1] = (g_band.values - self.df_g_band_m) / self.df_g_band_s
        features[2] = (r_band.values - self.df_r_band_m) / self.df_r_band_s
        features[3] = (i_band.values - self.df_i_band_m) / self.df_i_band_s
        features[4] = (z_band.values - self.df_z_band_m) / self.df_z_band_s
        features[5] = (u_g.values - self.u_g_m ) / self.u_g_s
        features[6] = (g_r.values - self.g_r_m ) / self.g_r_s
        features[7] = (r_i.values - self.r_i_m ) / self.r_i_s
        features[8] = (i_z.values - self.i_z_m ) / self.i_z_s
        features[9] = (z_u.values - self.z_u_m ) / self.z_u_s
        features[10] = u_band_std.values
        features[11] = g_band_std.values
        features[12] = r_band_std.values
        features[13] = i_band_std.values
        features[14] = z_band_std.values
        features[15] = (z.values - self.z_m ) / self.z_s
        features[16] = (M_i.values - self.M_i_m ) / self.M_i_s
        features[17] = (tau.values - self.tau_m ) / self.tau_s
        features[18] = (sigma.values - self.sigma_m ) / self.sigma_s
        features[19:] = np.tanh(hidden_representation[0].data.cpu().numpy())




        return features, M.values, z.values

    def __len__(self):
        return self.df.shape[0]



train_loader = torch.utils.data.DataLoader(BHDataset(folder, image_path, encoder_net_path, train=True, transform=data_transform, target_transform=target_transform),
                    batch_size = glo_batch_size, shuffle = True
                    )

if __name__ == '__main__':

    dset_classes_number = 1
    net = MLP(x_dim=29, h_dim1= 64, h_dim2=64, h_dim3=64, h_dim4=64, h_dim5=64, z_dim=1)
    loss_fn = nn.MSELoss(reduction='mean')

    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr = 2*1e-3)
    tb = SummaryWriter()

    best_accuracy = float("inf")
    model_number = 1


    for epoch in range(EPOCH):

        net.train()
        total_loss = 0.0
        test_loss = 0.0
        total_counter = 0
        test_counter = 0
        total_rms = 0

        for batch_idx, (data, BH_Mass, z) in enumerate(tqdm(train_loader, total = len(train_loader))):
            data, target = data.float(), BH_Mass.float()
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

            test_loader = torch.utils.data.DataLoader(BHDataset(folder, image_path, encoder_net_path, train=False, transform=data_transform, target_transform=target_transform),
                        batch_size = glo_batch_size, shuffle = True
                        )

            for batch_idx, (data, BH_Mass, z) in enumerate(test_loader):
                data, target = data.float(), BH_Mass.float()
                data, target = Variable(data).cuda(), Variable(target).cuda()

                #pred [batch, out_caps_num, out_caps_size, 1]
                pred = net(data)
                loss = loss_fn(pred, target)
                square_diff = (pred - target)
                total_rms += square_diff.std(dim=0)
                total_loss += loss.item()
                total_counter += 1

                # if batch_idx % test_num_batch == 0 and batch_idx != 0:
                #     tb.add_scalar('test_loss', loss.item())
                #     break

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
                torch.save(net, './saved_model/' + datetime_today + '_' + str(model_number) + '_'+ 'BHMass_hybrid_lc_image_enc_z.mdl')
                model_number += 1
                print("saved to " + "BHMass_hybrid_lc_image_enc_z.mdl" + " file.")

    tb.close()
