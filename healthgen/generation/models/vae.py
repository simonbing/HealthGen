"""
2021 Simon Bing, ETHZ, MPI IS
"""

from torch import nn
import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, seq_len=25, activation='tanh', dense_x_z=[],
                 dense_z_x=[], conv_x_z=[], conv_z_x={}, dropout_p=0, encoder='mlp', beta=1, data_mode='mask', mask_loss=False, device='cpu'):

        super().__init__()
        ### General parameters
        self.x_dim = x_dim
        self.y_dim = 1 # Needed for concatenation, can change if we use one-hot encoding
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        self.encoder = encoder
        self.data_mode = data_mode
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise SystemExit('Wrong activation type!')
        self.mask_loss = mask_loss
        self.device = device
        ### Inference
        self.dense_x_z = dense_x_z
        self.conv_x_z = conv_x_z
        ### Generation
        self.dense_z_x = dense_z_x
        self.conv_z_x = conv_z_x
        ### Beta
        self.beta = beta

        self.build()


    def build(self):
        ###################
        #### Inference ####
        ###################
        dic_layers = OrderedDict()
        if self.encoder == 'mlp':
            if len(self.dense_x_z) == 0:
                dim_x_z = self.x_dim+self.y_dim
                dic_layers['Identity'] = nn.Identity()
            else:
                dim_x_z = self.dense_x_z[-1]
                for n in range(len(self.dense_x_z)):
                    if n == 0:
                        dic_layers['linear'+str(n)] = nn.Linear(self.x_dim*self.seq_len+self.y_dim, self.dense_x_z[n])
                    else:
                        dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_z[n-1], self.dense_x_z[n])
                    dic_layers['activation' + str(n)] = self.activation
                    dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
            self.mlp_x_z = nn.Sequential(dic_layers)
            res_len = 1
        elif self.encoder == 'conv':
            dim_x_z = self.conv_x_z[-1]
            for n in range(len(self.conv_x_z)):
                if n == 0:
                    dic_layers['conv'+str(n)] = nn.Conv1d(self.x_dim, self.conv_x_z[n], kernel_size=4)
                else:
                    dic_layers['conv'+str(n)] = nn.Conv1d(self.conv_x_z[n-1], self.conv_x_z[n], kernel_size=4)
                dic_layers['avg_pool'+str(n)] = nn.AvgPool1d(kernel_size=2)
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
            self.conv_enc = nn.Sequential(dic_layers)
            res_len = 4

        self.z_mean = nn.Linear(dim_x_z*res_len, self.z_dim)
        self.z_logvar = nn.Linear(dim_x_z*res_len, self.z_dim)

        ######################
        #### Generation x ####
        ######################
        dic_layers = OrderedDict()
        if self.encoder == 'mlp':
            if len(self.dense_z_x) == 0:
                dim_z_x = self.z_dim+self.y_dim
                dic_layers['Identity'] = nn.Identity()
            else:
                dim_z_x = self.dense_z_x[-1]
                for n in range(len(self.dense_z_x)):
                    if n == 0:
                        dic_layers['linear'+str(n)] = nn.Linear(self.z_dim+self.y_dim, self.dense_z_x[n])
                    else:
                        dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
                    dic_layers['activation' + str(n)] = self.activation
                    dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
            self.mlp_z_x = nn.Sequential(dic_layers)
            if self.data_mode == 'feats':
                self.gen_x_mean = nn.Linear(dim_z_x, self.x_dim*self.seq_len)
                self.gen_x_logvar = nn.Linear(dim_z_x, self.x_dim*self.seq_len)
            elif self.data_mode == 'mask':
                self.gen_x_mean = nn.Sequential(nn.Linear(dim_z_x, self.x_dim*self.seq_len), nn.Sigmoid()) # For binary outputs
        elif self.encoder == 'conv':
            self.mlp_z_x = nn.Linear(self.z_dim+self.y_dim, dim_x_z*res_len)
            for n in range(len(self.conv_z_x)):
                if n == len(self.conv_z_x) - 1: # Last layer
                    dic_layers['upsample'+str(n)] = nn.Upsample(scale_factor=2, mode='linear')
                    dic_layers['conv'+str(n)] = nn.ConvTranspose1d(self.conv_z_x[n], self.x_dim, kernel_size=4)
                    if self.data_mode == 'feats':
                        pass # No activation for now
                    elif self.data_mode == 'mask':
                        dic_layers['activation' + str(n)] = nn.Sigmoid() # Bernoulli distribution in output
                    dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
                else:
                    dic_layers['upsample'+str(n)] = nn.Upsample(scale_factor=2, mode='linear')
                    dic_layers['conv'+str(n)] = nn.ConvTranspose1d(self.conv_z_x[n], self.conv_z_x[n+1], kernel_size=4)
                    dic_layers['activation' + str(n)] = self.activation
                    dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
            self.conv_dec = nn.Sequential(dic_layers)



    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, labels):
        if self.encoder == 'mlp':
            # Flatten input
            x_re = torch.reshape(x, (x.shape[0], -1))
            # Concatenate labels for conditioning
            x_cat = torch.cat((x_re, labels[:, :, 0]), 1)

            x_z = self.mlp_x_z(x_cat)
        elif self.encoder == 'conv':
            x = x.permute(0, 2, 1)
            x_z = self.conv_enc(x)
            # Flatten before final MLP
            x_z = torch.flatten(x_z, start_dim=1)

        z_mean = self.z_mean(x_z)
        z_logvar = self.z_logvar(x_z)
        z = self.reparameterization(z_mean, z_logvar)

        return z, z_mean, z_logvar


    def generation(self, z):
        if self.encoder == 'mlp':
            z_x = self.mlp_z_x(z)
            x_mean = self.gen_x_mean(z_x)
            if self.data_mode == 'feats':
                # x_logvar = self.gen_x_logvar(z_x)
                x_logvar = torch.log(0.1 * torch.ones_like(x_mean))
                return (x_mean, x_logvar)
            elif self.data_mode == 'mask':
                return x_mean
        elif self.encoder == 'conv':
            z_x = self.mlp_z_x(z)
            z_x = torch.reshape(z_x, (z_x.shape[0], self.conv_z_x[0], -1))
            x_mean = self.conv_dec(z_x)
            if self.data_mode == 'feats':
                x_logvar = torch.log(0.1 * torch.ones_like(x_mean))
                return (x_mean, x_logvar)
            elif self.data_mode == 'mask':
                return x_mean


    def forward(self, x, labels, compute_loss):
        # Need input: (batch_size, seq_len, x_dim)
        x = x.permute(0, 2, 1)
        # if self.encoder == 'mlp':
        #     # Flatten input
        #     x_re = torch.reshape(x, (x.shape[0], -1))
        #     # Concatenate labels for conditioning
        #     x_cat = torch.cat((x_re, labels[:,:,0]), 1)
        # if self.encoder == 'conv':
        #     # Need other shape for 1D conv
        #     x = x.permute(0, 2, 1)

        # Inference
        z, z_mean, z_logvar = self.inference(x, labels)
        # Concatenate labels for conditional generation
        z_cat = torch.cat((z, labels[:,:,0]), 1)
        # Generation
        x_hat = self.generation(z_cat)
        if self.encoder == 'mlp':
            # Reshape to original shape
            x_hat = [torch.reshape(x, (-1, self.seq_len, self.x_dim)) for x in x_hat]
        elif self. encoder == 'conv':
            x_hat = [x.permute(0, 2, 1) for x in x_hat]

        # Calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, x_hat, z_mean,
                                                           z_logvar, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # Reshape back to original shape
        x_hat = [x.permute(0, 2, 1) for x in x_hat]

        return x_hat[0] # Return mean for reconstruction


    def get_loss(self, x, x_hat, z_mean, z_logvar, beta=1):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        if self.data_mode == 'mask':
            # Bernoulli LL for binary masks
            eps = 1e-6
            x_hat = torch.clamp(x_hat[0], eps, 1-eps)
            loss_recon = -(x * torch.log(x_hat) + (1 - x) * torch.log(1 - x_hat))
        elif self.data_mode =='feats':
            x_hat_mean = x_hat[0]
            x_hat_logvar = x_hat[1]
            const_log_pdf = (- 0.5 * np.log(2 * np.pi))
            loss_recon = -const_log_pdf + 0.5 * x_hat_logvar + torch.square(x - x_hat_mean) / (2 * torch.exp(x_hat_logvar))

        loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar.exp() - z_mean.pow(2) + 1)

        loss_recon = torch.sum(loss_recon) / batch_size
        loss_KLD = loss_KLD / batch_size

        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD


    def generate_synth(self, labels, seq_len, det_gen=False, batch_size=64):
        # Make labels dataloader
        print(F'Device in vae.py: {self.device}')
        labels_loader = DataLoader(torch.Tensor(labels), batch_size=batch_size)

        # Outer batch loop
        x_batched = []
        for label_batch in labels_loader:
            batch_len = len(label_batch)
            label_batch = label_batch.to(self.device)

            # Sample from standard Gaussian Prior
            z = torch.randn(batch_len, self.z_dim)
            z = z.to(self.device)
            # Concatenate labels for conditional generation
            z_cat = torch.cat((z, torch.unsqueeze(label_batch, 1)), 1)
            # z_cat = z_cat.to(self.device)

            # Generate
            x_mean = self.generation(z_cat)
            if self.encoder == 'mlp':
                x_mean = [torch.reshape(x, (-1, self.seq_len, self.x_dim)) for x in x_mean]
            elif self.encoder == 'conv':
                x_mean = [x.permute(0, 2, 1) for x in x_mean]
            x_mean = torch.cat(x_mean)

            if self.data_mode == 'feats':
                # x_mean_flat = torch.flatten(x_mean[0], start_dim=1)
                # x_logvar_flat = torch.flatten(x_mean[1], start_dim=1)
                # x_distr = MultivariateNormal(loc=x_mean_flat, covariance_matrix=torch.diag_embed(torch.exp(x_logvar_flat)))
                # x = x_distr.sample()
                # x = torch.reshape(x, x_mean[0].shape)
                x = x_mean[0]
            elif self.data_mode == 'mask':
                # Sample actual values from Bernoulli distribution
                x_distr = Bernoulli(probs=x_mean)
                x = x_distr.sample().int()
            # Or: threshold probabilities without sampling
            # x = torch.round(x_mean)


            x_batched.append(x)

        X_synth = torch.cat(x_batched, dim=0)
        X_synth = X_synth.permute(0, 2, 1)

        return X_synth.cpu().detach().numpy(), labels