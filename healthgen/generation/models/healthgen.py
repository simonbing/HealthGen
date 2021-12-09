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

class HealthGen(nn.Module):
    def __init__(self, x_dim, y_dim, v_dim, z_dim, seq_len, activation='tanh', dense_x_v=[256,128],
                 dense_x_h=[], dense_hx_g=[], dense_gz_z=[128,128], dim_RNN_h=128, num_RNN_h=1, dim_RNN_g=128, num_RNN_g=1,
                 dense_hz_z=[64,32], dense_hz_x=[128,128], dense_v_m=[128,256], dropout_p=0, beta=1, device='cpu'):
        super().__init__()

        ### General parameters
        self.x_dim = x_dim
        self.y_dim = y_dim  # Needed for concatenation, can change if we use one-hot encoding
        self.v_dim = v_dim
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Inference
        self.dense_x_v = dense_x_v
        self.dense_x_h = dense_x_h
        self.dense_hx_g = dense_hx_g
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        self.dense_gz_z = dense_gz_z
        ### Generation
        self.dense_hz_z = dense_hz_z
        self.dense_hz_x = dense_hz_x
        self.dense_v_m = dense_v_m
        ### Beta
        self.beta = beta

        self.build()


    def build(self):
        ###################
        ### Inference v ###
        ###################
        dic_layers = OrderedDict()
        # Only implementing MLP encoder for now
        dim_x_v = self.dense_x_v[-1]
        for n in range(len(self.dense_x_v)):
            if n == 0:
                dic_layers[F'linear{n}'] = nn.Linear(2*self.x_dim*self.seq_len+self.y_dim,
                                                     self.dense_x_v[n])
            else:
                dic_layers[F'linear{n}'] = nn.Linear(self.dense_x_v[n-1], self.dense_x_v[n])
            dic_layers[F'activation{n}'] = self.activation
            dic_layers[F'dropout{n}'] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_v = nn.Sequential(dic_layers)

        self.v_mean = nn.Linear(dim_x_v, self.v_dim)
        self.v_logvar = nn.Linear(dim_x_v, self.v_dim)

        ###################
        ### Inference z ###
        ###################
        # Deterministic forward recurrance (h_t)
        dic_layers = OrderedDict()
        if len(self.dense_x_h) == 0:
            dim_x_h = self.x_dim + self.v_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_h = self.dense_x_h[-1]
            for n in range(len(self.dense_x_h)):
                if n == 0:
                    dic_layers[F'linear{n}'] = nn.Linear(self.x_dim+self.v_dim, self.dense_x_h[n])
                else:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dense_x_h[n-1], self.dense_x_h[n])
                dic_layers[F'activation{n}'] = self.activation
                dic_layers[F'dropout{n}'] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_h = nn.Sequential(dic_layers)
        self.rnn_h = nn.GRU(dim_x_h, self.dim_RNN_h, self.num_RNN_h, batch_first=True)

        # Backward recurrance (g_t)
        dic_layers = OrderedDict()
        if len(self.dense_hx_g) == 0:
            dim_hx_g = self.dim_RNN_h + self.x_dim + self.v_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hx_g = self.dense_hx_g[-1]
            for n in range(len(self.dense_hx_g)):
                if n == 0:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dim_RNN_h+self.x_dim+self.v_dim, self.dense_hx_g[n])
                else:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dense_hx_g[n-1], self.dense_hx_g[n])
                dic_layers[F'activation{n}'] = self.activation
                dic_layers[F'dropout{n}'] = nn.Dropout(p=self.dropout_p)
        self.mlp_hx_g = nn.Sequential(dic_layers)
        self.rnn_g = nn.GRU(dim_hx_g, self.dim_RNN_g, self.num_RNN_g, batch_first=True)

        # Prior dynamics
        dic_layers = OrderedDict()
        if len(self.dense_gz_z) == 0:
            dim_gz_z = self.dim_RNN_g + self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_gz_z = self.dense_gz_z[-1]
            for n in range(len(self.dense_gz_z)):
                if n == 0:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dim_RNN_g+self.z_dim, self.dense_gz_z[n])
                else:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dense_gz_z[n-1], self.dense_gz_z[n])
                dic_layers[F'activation{n}'] = self.activation
                dic_layers[F'dropout{n}'] = nn.Dropout(p=self.dropout_p)
        self.mlp_gz_z = nn.Sequential(dic_layers)

        self.z_mean = nn.Linear(dim_gz_z, self.z_dim)
        self.z_logvar = nn.Linear(dim_gz_z, self.z_dim)

        ######################
        #### Generation z ####
        ######################
        dic_layers = OrderedDict()
        if len(self.dense_hz_z) == 0:
            dim_hz_z = self.dim_RNN_h + self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_z = self.dense_hz_z[-1]
            for n in range(len(self.dense_hz_z)):
                if n == 0:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dim_RNN_h+self.z_dim, self.dense_hz_z[n])
                else:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dense_hz_z[n-1], self.dense_hz_z[n])
                dic_layers[F'activation{n}'] = self.activation
                dic_layers[F'dropout{n}'] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_z = nn.Sequential(dic_layers)

        self.z_prior_mean = nn.Linear(dim_hz_z, self.z_dim)
        self.z_prior_logvar = nn.Linear(dim_hz_z, self.z_dim)

        ######################
        #### Generation x ####
        ######################
        dic_layers = OrderedDict()
        if len(self.dense_hz_x) == 0:
            dim_hz_x = self.dim_RNN_h + self.z_dim + self.v_dim + self.y_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_x = self.dense_hz_x[-1]
            for n in range(len(self.dense_hz_x)):
                if n == 0:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dim_RNN_h+self.z_dim+self.v_dim+self.y_dim, self.dense_hz_x[n])
                else:
                    dic_layers[F'linear{n}'] = nn.Linear(self.dense_hz_x[n-1], self.dense_hz_x[n])
                dic_layers[F'activation{n}'] = self.activation
                dic_layers[F'dropout{n}'] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_x = nn.Sequential(dic_layers)

        self.gen_x_mean = nn.Linear(dim_hz_x, self.x_dim)
        self.gen_x_logvar = nn.Linear(dim_hz_x, self.x_dim)

        ####################
        ### Generation m ###
        ####################
        dic_layers = OrderedDict()
        dim_v_m = self.dense_v_m[-1]
        for n in range(len(self.dense_v_m)):
            if n == 0:
                dic_layers[F'linear{n}'] = nn.Linear(self.v_dim+self.y_dim, self.dense_v_m[n])
            else:
                dic_layers[F'linear{n}'] = nn.Linear(self.dense_v_m[n-1], self.dense_v_m[n])
            dic_layers[F'activation{n}'] = self.activation
            dic_layers[F'dropout{n}'] = nn.Dropout(p=self.dropout_p)
        self.mlp_v_m = nn.Sequential(dic_layers)

        self.gen_m_mean = nn.Sequential(nn.Linear(dim_v_m, self.x_dim*self.seq_len), nn.Sigmoid())


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)

    def deterministic_h(self, x_tm1):
        x_h = self.mlp_x_h(x_tm1)
        h, _ = self.rnn_h(x_h)

        return h


    def inference(self, feats, masks, labels):
        batch_size = feats.shape[0]
        seq_len = feats.shape[2]

        # 1. Static variable inference (x_t, m_t -> v)
        x_cat = torch.cat((feats, masks), dim=1)
        # x_cat = masks # For experimental inference using only masks, not feats
        x_flat = torch.flatten(x_cat, start_dim=1)
        # Concatenate labels for conditioning
        x_cond = torch.cat((x_flat, labels[:,:,0]), dim=1)
        x_v = self.mlp_x_v(x_cond)

        v_mean = self.v_mean(x_v)
        v_logvar = self.v_logvar(x_v)
        v = self.reparameterization(v_mean, v_logvar)

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros(batch_size, seq_len, self.z_dim).to(self.device)
        z_logvar = torch.zeros(batch_size, seq_len, self.z_dim).to(self.device)
        z = torch.zeros(batch_size, seq_len, self.z_dim).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)

        # 2. Dynamic variable inference (x_t -> z_t)
        # Repeat v for all time steps
        v_tile = torch.tile(v.unsqueeze(1), (1, seq_len, 1))
        # 2.1 x_tm1 -> h_t
        feats = feats.permute(0, 2, 1)
        x_0 = torch.zeros(batch_size, 1, self.x_dim).to(self.device)
        x_tm1 = torch.cat((x_0, feats[:, :-1, :]), dim=1)
        x_tm1 = torch.cat((x_tm1, v_tile), dim=2) # For additional conditioning
        h = self.deterministic_h(x_tm1)
        # 2.2 x_t, h_t -> g_t
        hx_g = torch.cat((h, feats, v_tile), dim=2)
        hx_g = self.mlp_hx_g(hx_g)
        g_inv, _ = self.rnn_g(torch.flip(hx_g, [1]))
        g = torch.flip(g_inv, [1])
        # 2.3 g_t, z_tm1 -> z_t
        for t in range(seq_len):
            gz_z = torch.cat((g[:,t,:], z_t), dim=-1)
            gz_z = self.mlp_gz_z(gz_z)
            z_mean[:,t,:] = self.z_mean(gz_z)
            z_logvar[:,t,:] = self.z_logvar(gz_z)
            z_t = self.reparameterization(z_mean[:,t,:], z_logvar[:,t,:])
            z[:,t,:] = z_t

        return v, v_mean, v_logvar, z, z_mean, z_logvar, h


    def generation_z(self, h, z):
        batch_size = z.shape[0]

        z_0 = torch.zeros(batch_size, 1, self.z_dim).to(self.device)
        z_tm1 = torch.cat((z_0, z[:, :-1, :]), dim=1)

        hz_z = torch.cat((h, z_tm1), dim=-1)
        hz_z = self.mlp_hz_z(hz_z)
        z_prior_mean = self.z_prior_mean(hz_z)
        z_prior_logvar = self.z_prior_logvar(hz_z)

        return z_prior_mean, z_prior_logvar


    def generation_x(self, z, v, h, labels):
        seq_len = z.shape[1]

        # Feature generation
        labels = labels.permute(0, 2, 1)
        # Repeat v for all timesteps
        v_tile = torch.tile(v.unsqueeze(1), (1, seq_len, 1))
        # Concatenate for conditional generation
        hz_cat = torch.cat((h, z, v_tile, labels), dim=-1)
        hz_x = self.mlp_hz_x(hz_cat)

        x_mean = self.gen_x_mean(hz_x)
        x_logvar = self.gen_x_logvar(hz_x)

        # Mask generation
        v_cat = torch.cat((v, labels[:, 0, :]), dim=1)
        v_m = self.mlp_v_m(v_cat)

        m_mean = self.gen_m_mean(v_m)
        m_mean = m_mean.view(-1, self.x_dim, self.seq_len)

        return x_mean.permute(0,2,1), x_logvar.permute(0,2,1), m_mean



    def forward(self, x, labels, compute_loss):
        x_feats = x[:, 0, ...]
        x_masks = x[:, 1, ...]

        x_cat = torch.cat((x_feats, x_masks), dim=1)

        # Inference
        v, v_mean, v_logvar, z, z_mean, z_logvar, h = self.inference(x_feats, x_masks, labels)

        # Generation
        z_prior_mean, z_prior_logvar = self.generation_z(h, z)
        x_hat_mean, x_hat_logvar, m_hat_mean = self.generation_x(z, v, h, labels)

        # Compute loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x_feats, x_masks, x_hat_mean,
                                                           x_hat_logvar, m_hat_mean, z_mean,
                                                           z_logvar, z_prior_mean, z_prior_logvar,
                                                           v_mean, v_logvar, self.beta)

            self.loss = (loss_tot, loss_recon, loss_KLD)

        x_hat = torch.stack((x_hat_mean, m_hat_mean), dim=1)

        return x_hat


    def get_loss(self, x_feats, x_masks, x_hat_feats_mean, x_hat_feats_logvar,
                 x_hat_masks_mean, z_mean, z_logvar, z_prior_mean, z_prior_logvar,
                 v_mean, v_logvar, beta):
        batch_size = x_feats.shape[0]

        # Recon loss features
        const_log_pdf = (- 0.5 * np.log(2 * np.pi))
        loss_recon_feats = -const_log_pdf + 0.5 * x_hat_feats_logvar \
                           + torch.square(x_feats - x_hat_feats_mean) \
                           / (2 * torch.exp(x_hat_feats_logvar))
        # Mask loss
        loss_recon_feats = torch.masked_select(loss_recon_feats, x_masks.bool())
        loss_recon_feats = torch.sum(loss_recon_feats) / batch_size

        # Recon loss masks
        eps = 1e-6
        x_hat_masks_mean = torch.clamp(x_hat_masks_mean, eps, 1 - eps)
        loss_recon_masks = -(x_masks * torch.log(x_hat_masks_mean)
                             + (1 - x_masks) * torch.log(1 - x_hat_masks_mean))
        loss_recon_masks = torch.sum(loss_recon_masks) / batch_size

        loss_recon = loss_recon_feats + loss_recon_masks

        # KL loss dynamic latents
        loss_KLD_dyn = -0.5 * torch.sum(z_logvar - z_prior_logvar - torch.div(z_logvar.exp()
                                                                          + (z_mean - z_prior_mean).pow(2),
                                                                          z_prior_logvar.exp()))
        loss_KLD_dyn = loss_KLD_dyn / batch_size

        # KL loss static latents
        loss_KLD_stat = -0.5 * torch.sum(v_logvar - v_logvar.exp() - v_mean.pow(2) + 1)
        loss_KLD_stat = loss_KLD_stat / batch_size

        loss_KLD = loss_KLD_dyn + loss_KLD_stat

        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD

    def generate_synth(self, labels, seq_len, det_gen=False, batch_size=64):
        # Make labels dataloader
        # TODO: handle cases where we condition on static data
        labels_loader = DataLoader(torch.Tensor(labels), batch_size=batch_size)

        # Outer batch loop
        x_batched = []
        for label_batch in labels_loader:
            batch_len = len(label_batch)
            label_batch = label_batch.to(self.device)

            # Masks generation
            v = torch.randn(batch_len, self.v_dim)
            v = v.to(self.device)
            if len(label_batch.shape) == 1:
                label_batch_v = label_batch.unsqueeze(1)
            else:
                label_batch_v = label_batch
            v_cat = torch.cat((v, label_batch_v), dim=1)
            v_m = self.mlp_v_m(v_cat)
            m_mean = self.gen_m_mean(v_m)
            m_mean = m_mean.view(-1, self.x_dim, self.seq_len)
            m_distr = Bernoulli(probs=m_mean)
            x_masks = m_distr.sample().int()

            # Initialize h_0
            h = torch.zeros(batch_len, 1, self.dim_RNN_h).to(self.device)
            # Initialize RNN hidden state
            hidden_state = torch.zeros(self.rnn_h.num_layers, batch_len,
                                       self.rnn_h.hidden_size).to(self.device)
            # Initialize initial latent state z_tm1 (sample from standard Gaussian prior)
            z_tm1 = torch.randn(batch_len, 1, self.z_dim).to(self.device)

            # Inner loop over time steps
            if len(label_batch.shape) == 1:
                label_batch = torch.unsqueeze(torch.unsqueeze(label_batch, 1), -1)
            else:
                label_batch = torch.unsqueeze(label_batch, 1)
            x_feats_seq = []
            for t in range(seq_len):
                # Get next z from previous z and h
                z_cat = torch.cat((h, z_tm1), dim=-1)
                hz_z = self.mlp_hz_z(z_cat)
                z_mean = self.z_prior_mean(hz_z)
                z_logvar = self.z_prior_logvar(hz_z)
                z = torch.normal(z_mean, torch.sqrt(torch.exp(z_logvar)))

                # Conditioning step
                hz_cat = torch.cat((h, z, v.unsqueeze(1), label_batch), dim=-1)
                # Generate x from hz_cat
                hz_x = self.mlp_hz_x(hz_cat)
                x_mean = self.gen_x_mean(hz_x).permute(0, 2, 1)
                x_logvar = self.gen_x_logvar(hz_x).permute(0, 2, 1)
                x_mean_flat = torch.flatten(x_mean, start_dim=1)
                x_logvar_flat = torch.flatten(x_logvar, start_dim=1)
                x_distr = MultivariateNormal(loc=x_mean_flat,
                                             covariance_matrix=torch.diag_embed(torch.exp(x_logvar_flat)))
                x_feats = x_distr.sample()
                x_feats = x_feats.view(x_mean.shape)

                # Get next RNN state from generated x (and generated v)
                # x_h = x_feats.permute(0,2,1)
                x_h = torch.cat((x_feats.permute(0,2,1), v.unsqueeze(1)), dim=2) # For additional conditioning
                x_h = self.mlp_x_h(x_h)
                h, hidden_state = self.rnn_h(x_h, hidden_state)

                z_tm1 = z

                x_feats_seq.append(x_feats)
            x_feats_cat = torch.cat(x_feats_seq, dim=2)
            x = torch.cat((x_feats_cat, x_masks), dim=1)

            x_batched.append(x)

        X_synth = torch.cat(x_batched, dim=0)

        return X_synth.cpu().detach().numpy(), labels
