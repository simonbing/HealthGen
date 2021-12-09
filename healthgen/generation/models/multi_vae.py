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

class MultiVAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, seq_len=25, activation='tanh', dense_x_z=[],
                 dense_z_x=[], conv_x_z=[], conv_z_x={}, dropout_p=0, encoder='mlp',
                 decoder='mlp', beta=1, data_mode='mask', mask_loss=False, device='cpu'):

        super().__init__()
        ### General parameters
        self.x_dim = x_dim
        self.y_dim = 1 # Needed for concatenation, can change if we use one-hot encoding
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        self.encoder = encoder
        self.decoder = decoder
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
            dim_x_z = self.dense_x_z[-1]
            for n in range(len(self.dense_x_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(2*self.x_dim*self.seq_len+self.y_dim, self.dense_x_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_z[n-1], self.dense_x_z[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
            self.mlp_x_z = nn.Sequential(dic_layers)
        elif self.encoder == 'conv':
            raise NotImplementedError

        self.z_mean = nn.Linear(dim_x_z, self.z_dim)
        self.z_logvar = nn.Linear(dim_x_z, self.z_dim)

        ######################
        #### Generation x ####
        ######################
        # feats decoder
        dic_layers = OrderedDict()
        if self.decoder == 'mlp':
            dim_z_x = self.dense_z_x[-1]
            for n in range(len(self.dense_z_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim+self.y_dim, self.dense_z_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
            self.mlp_z_x_feats = nn.Sequential(dic_layers)
            self.gen_x_mean_feats = nn.Linear(dim_z_x, self.x_dim*self.seq_len)
            self.gen_x_logvar_feats = nn.Linear(dim_z_x, self.x_dim*self.seq_len)
        elif self.decoder == 'conv':
            res_len = 4
            self.mlp_z_x_feats = nn.Linear(self.z_dim+self.y_dim, self.conv_x_z[-1]*res_len)
            for n in range(len(self.conv_z_x)):
                if n == len(self.conv_z_x) - 1: # Last layer
                    dic_layers['upsample'+str(n)] = nn.Upsample(scale_factor=2, mode='linear')
                    dic_layers['conv'+str(n)] = nn.ConvTranspose1d(self.conv_z_x[n], self.x_dim, kernel_size=4)
                else:
                    dic_layers['upsample'+str(n)] = nn.Upsample(scale_factor=2, mode='linear')
                    dic_layers['conv'+str(n)] = nn.ConvTranspose1d(self.conv_z_x[n], self.conv_z_x[n+1], kernel_size=4)
                    dic_layers['activation' + str(n)] = self.activation
                    dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
            self.conv_dec = nn.Sequential(dic_layers)

        # masks decoder
        dic_layers = OrderedDict()
        dim_z_x = self.dense_z_x[-1]
        for n in range(len(self.dense_z_x)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(self.z_dim+self.y_dim, self.dense_z_x[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
            dic_layers['activation' + str(n)] = self.activation
            dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_z_x_masks = nn.Sequential(dic_layers)
        self.gen_x_mean_masks = nn.Sequential(nn.Linear(dim_z_x, self.x_dim * self.seq_len), nn.Sigmoid())  # For binary outputs


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, labels):
        if self.encoder == 'mlp':
            # Flatten input
            x_flat = torch.flatten(x, start_dim=1)
            # Concatenate labels for conditioning
            x_labels_cat = torch.cat((x_flat, labels[:, :, 0]), 1)

            x_z = self.mlp_x_z(x_labels_cat)
        elif self.encoder == 'conv':
            raise NotImplementedError

        z_mean = self.z_mean(x_z)
        z_logvar = self.z_logvar(x_z)
        z = self.reparameterization(z_mean, z_logvar)

        return z, z_mean, z_logvar


    def generation(self, z):
        # feats
        z_x_feats = self.mlp_z_x_feats(z)
        if self.decoder == 'mlp':
            x_mean_feats = self.gen_x_mean_feats(z_x_feats)
            x_logvar_feats = self.gen_x_logvar_feats(z_x_feats)
            # Reshape to original shape
            x_mean_feats = x_mean_feats.view(-1, self.x_dim, self.seq_len)
            x_logvar_feats = x_logvar_feats.view(-1, self.x_dim, self.seq_len)
        elif self.decoder == 'conv':
            z_x_feats = z_x_feats.view(z_x_feats.shape[0], self.conv_z_x[0], -1)
            x_mean_feats = self.conv_dec(z_x_feats)
            x_logvar_feats = torch.log(0.1 * torch.ones_like(x_mean_feats))

        # masks
        z_x_masks = self.mlp_z_x_masks(z)
        x_mean_masks = self.gen_x_mean_masks(z_x_masks)
        # Reshape to original shape
        x_mean_masks = x_mean_masks.view(-1, self.x_dim, self.seq_len)

        return x_mean_feats, x_logvar_feats, x_mean_masks


    def forward(self, x, labels, compute_loss):
        x_feats = x[:,0,...]
        x_masks = x[:,1,...]

        x_cat = torch.cat((x_feats, x_masks), dim=1)

        # Inference
        z, z_mean, z_logvar = self.inference(x_cat, labels)

        # Concatenate labels for conditional generation
        z_labels_cat = torch.cat((z, labels[:, :, 0]), 1)

        # Generation
        x_hat_feats_mean, x_hat_feats_logvar, x_hat_masks_mean = self.generation(z_labels_cat)

        # Compute loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x_feats, x_masks, x_hat_feats_mean,
                                                           x_hat_feats_logvar, x_hat_masks_mean,
                                                           z_mean, z_logvar, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # Stack features and masks again
        x_hat = torch.stack((x_hat_feats_mean, x_hat_masks_mean), dim=1)

        return x_hat


    def get_loss(self, x_feats, x_masks, x_hat_feats_mean, x_hat_feats_logvar, x_hat_masks_mean, z_mean, z_logvar, beta=1):
        batch_size = x_feats.shape[0]

        # Recon loss features
        const_log_pdf = (- 0.5 * np.log(2 * np.pi))
        loss_recon_feats = -const_log_pdf + 0.5 * x_hat_feats_logvar + torch.square(x_feats - x_hat_feats_mean) / (2 * torch.exp(x_hat_feats_logvar))
        # Mask loss
        loss_recon_feats = torch.masked_select(loss_recon_feats, x_masks.bool())
        loss_recon_feats = torch.sum(loss_recon_feats) / batch_size
        # Recon loss masks
        eps = 1e-6
        x_hat_masks_mean = torch.clamp(x_hat_masks_mean, eps, 1 - eps)
        loss_recon_masks = -(x_masks * torch.log(x_hat_masks_mean) + (1 - x_masks) * torch.log(1 - x_hat_masks_mean))
        loss_recon_masks = torch.sum(loss_recon_masks) / batch_size

        loss_recon = loss_recon_feats + loss_recon_masks

        # KL loss
        loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar.exp() - z_mean.pow(2) + 1)
        loss_KLD = loss_KLD / batch_size

        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD

    def generate_synth(self, labels, seq_len, det_gen=False, batch_size=64):
        # Make labels dataloader
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

            # Generate
            x_feats_mean, x_feats_logvar, x_masks_mean = self.generation(z_cat)
            # feats
            x_feats_mean_flat = torch.flatten(x_feats_mean, start_dim=1)
            x_feats_logvar_flat = torch.flatten(x_feats_logvar, start_dim=1)
            x_feats_distr = MultivariateNormal(loc=x_feats_mean_flat,
                                               covariance_matrix=torch.diag_embed(torch.exp(x_feats_logvar_flat)))
            x_feats = x_feats_distr.sample()
            x_feats = x_feats.view(x_feats_mean.shape)
            # x_feats = x_feats_mean # Uncomment if we just want to take mean
            # masks
            x_masks_distr = Bernoulli(probs=x_masks_mean)
            x_masks = x_masks_distr.sample().int()

            # Stack features and masks
            x = torch.cat((x_feats, x_masks), dim=1)

            x_batched.append(x)

        X_synth = torch.cat(x_batched, dim=0)

        return X_synth.cpu().detach().numpy(), labels


