#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “A Disentanagled Recognition and Nonlinear Dynamics Model for Unsupervised Learning” NIPS, 2017, Macro Fraccaro et al.

Not include:
- different learning target (alpha first, then KF params, finally total params)
- no imputation
"""

import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
from collections import OrderedDict
import wandb


def build_KVAE(cfg, device='cpu'):
    ### Load special parameters for KVAE
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    a_dim = cfg.getint('Network', 'a_dim')
    z_dim = cfg.getint('Network', 'z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    scale_reconstruction = cfg.getfloat('Network', 'scale_reconstruction')
    # VAE
    dense_x_a = [] if cfg.get('Network', 'dense_x_a') == '' else [int(i) for i
                                                                  in cfg.get(
            'Network', 'dense_x_a').split(',')]
    dense_a_x = [] if cfg.get('Network', 'dense_a_x') == '' else [int(i) for i
                                                                  in cfg.get(
            'Network', 'dense_a_x').split(',')]
    # LGSSM
    init_kf_mat = cfg.getfloat('Network', 'init_kf_mat')
    noise_transition = cfg.getfloat('Network', 'noise_transition')
    noise_emission = cfg.getfloat('Network', 'noise_emission')
    init_cov = cfg.getfloat('Network', 'init_cov')
    # Dynamics
    K = cfg.getint('Network', 'K')
    dim_RNN_alpha = cfg.getint('Network', 'dim_RNN_alpha')
    num_RNN_alpha = cfg.getint('Network', 'num_RNN_alpha')
    # Training set
    scheduler_training = cfg.getboolean('Training', 'scheduler_training')
    only_vae_epochs = cfg.getint('Training', 'only_vae_epochs')
    kf_update_epochs = cfg.getint('Training', 'kf_update_epochs')

    # Build model
    model = KVAE_miss(x_dim=x_dim, a_dim=a_dim, z_dim=z_dim, activation=activation,
                 dense_x_a=dense_x_a, dense_a_x=dense_a_x,
                 init_kf_mat=init_kf_mat, noise_transition=noise_transition,
                 noise_emission=noise_emission, init_cov=init_cov,
                 K=K, dim_RNN_alpha=dim_RNN_alpha, num_RNN_alpha=num_RNN_alpha,
                 dropout_p=dropout_p, scale_reconstruction=scale_reconstruction,
                 device=device).to(device)

    return model


class KVAE_miss(nn.Module):

    def __init__(self, x_dim, m_dim, u_dim=1, a_dim=8, z_dim=4, activation='tanh',
                 dense_x_a=[128, 128], dense_a_x=[128, 128],
                 init_kf_mat=0.05, noise_transition=0.08, noise_emission=0.03,
                 init_cov=20,
                 K=3, dim_RNN_alpha=50, num_RNN_alpha=1,
                 dropout_p=0, scale_reconstruction=1, device='cpu', use_smoothed_a=False,
                 sample_m=False, learn_scale=False):

        super().__init__()
        ## General parameters
        self.x_dim = x_dim
        self.m_dim = m_dim
        self.y_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.u_dim = a_dim + u_dim
        # Training
        self.dropout_p = dropout_p
        self.scale_reconstruction = scale_reconstruction
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemError('Wrong activation type')
        self.device = device
        self.use_smoothed_a = use_smoothed_a
        self.sample_m = sample_m
        self.learn_scale = learn_scale
        # VAE
        self.dense_x_a = dense_x_a
        self.dense_a_x = dense_a_x
        # LGSSM
        self.init_kf_mat = init_kf_mat
        self.noise_transition = noise_transition
        self.noise_emission = noise_emission
        self.init_cov = init_cov
        # Dynamics params (alpha)
        self.K = K
        self.dim_RNN_alpha = dim_RNN_alpha
        self.num_RNN_alpha = num_RNN_alpha

        self.build()

    def build(self):

        #############
        #### VAE ####
        #############
        # 1. Inference of a_t
        dic_layers = OrderedDict()
        if len(self.dense_x_a) == 0:
            dim_x_a = self.x_dim + self.m_dim
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_x_a = self.dense_x_a[-1]
            for n in range(len(self.dense_x_a)):
                if n == 0:
                    dic_layers["linear" + str(n)] = nn.Linear(self.x_dim + self.m_dim,
                                                              self.dense_x_a[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dense_x_a[n - 1], self.dense_x_a[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_a = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_x_a, self.a_dim)
        self.inf_logvar = nn.Linear(dim_x_a, self.a_dim)
        # 2. Generation of x_t
        dic_layers = OrderedDict()
        if len(self.dense_a_x) == 0:
            dim_a_x = self.a_dim
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_a_x = self.dense_a_x[-1]
            for n in range(len(self.dense_x_a)):
                if n == 0:
                    dic_layers["linear" + str(n)] = nn.Linear(self.a_dim,
                                                              self.dense_a_x[n])
                else:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        self.dense_a_x[n - 1], self.dense_a_x[n])
                dic_layers["activation" + str(n)] = self.activation
                dic_layers["dropout" + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_a_x = nn.Sequential(dic_layers)
        # self.gen_mean = nn.Linear(dim_a_x, self.x_dim)
        # self.gen_logvar = nn.Linear(dim_a_x, self.x_dim)
        self.gen_m_mean = nn.Sequential(nn.Linear(dim_a_x, self.m_dim), nn.Sigmoid())

        ###############
        #### LGSSM ####
        ###############
        # Initializers for LGSSM variables, torch.tensor(), enforce torch.float32 type
        # A is an identity matrix
        # B and C are randomly sampled from a Gaussian
        # Q and R are isotroipic covariance matrices
        # z = Az + Bu
        # a = Cz
        self.A = torch.tensor(
            np.array([np.eye(self.z_dim) for _ in range(self.K)]),
            dtype=torch.float32, requires_grad=True,
            device=self.device)  # (K, z_dim. z_dim,)
        self.B = torch.tensor(np.array(
            [self.init_kf_mat * np.random.randn(self.z_dim, self.u_dim) for _ in
             range(self.K)]), dtype=torch.float32, requires_grad=True,
                              device=self.device)  # (K, z_dim, u_dim)
        self.C = torch.tensor(np.array(
            [self.init_kf_mat * np.random.randn(self.a_dim, self.z_dim) for _ in
             range(self.K)]), dtype=torch.float32, requires_grad=True,
                              device=self.device)  # (K, a_dim, z_dim)
        self.Q = self.noise_transition * torch.eye(self.z_dim).to(
            self.device)  # (z_dim, z_dim)
        self.R = self.noise_emission * torch.eye(self.a_dim).to(
            self.device)  # (a_dim, a_dim)
        self._I = torch.eye(self.z_dim).to(self.device)  # (z_dim, z_dim)

        ###############
        #### Alpha ####
        ###############
        self.a_init = torch.zeros((1, self.a_dim), requires_grad=True,
                                  device=self.device)  # (bs, a_dim)
        self.rnn_alpha = nn.LSTM(self.a_dim, self.dim_RNN_alpha,
                                 self.num_RNN_alpha, bidirectional=False, batch_first=True)
        self.mlp_alpha = nn.Sequential(nn.Linear(self.dim_RNN_alpha, self.K),
                                       nn.Softmax(dim=-1))

        ############################
        #### Scheduler Training ####
        ############################
        self.scale = torch.tensor(0.0)
        if self.learn_scale:
            self.scale = nn.Parameter(self.scale)

        self.A = nn.Parameter(self.A)
        self.B = nn.Parameter(self.B)
        self.C = nn.Parameter(self.C)
        self.a_init = nn.Parameter(self.a_init)
        kf_params = [self.A, self.B, self.C, self.a_init]

        # self.iter_kf = (i for i in kf_params)
        # self.iter_vae = self.concat_iter(self.mlp_x_a.parameters(),
        #                                  self.inf_mean.parameters(),
        #                                  self.inf_logvar.parameters(),
        #                                  self.mlp_a_x.parameters(),
        #                                  self.gen_logvar.parameters())
        # self.iter_alpha = self.concat_iter(self.rnn_alpha.parameters(),
        #                                    self.mlp_alpha.parameters())
        # self.iter_vae_kf = self.concat_iter(self.iter_vae, self.iter_kf)
        # self.iter_all = self.concat_iter(self.iter_kf, self.iter_vae,
        #                                  self.iter_alpha)

        # EXPERIMENTAL CONCATENATION
        # self.iter_kf = list(self.A) + list(self.B) + list(self.C) + list(self.a_init)
        self.iter_kf = kf_params
        self.iter_vae = list(self.mlp_x_a.parameters()) + list(self.inf_mean.parameters()) \
                        + list(self.inf_logvar.parameters()) + list(self.mlp_a_x.parameters()) \
                        + list(self.gen_m_mean.parameters())
                        # + list(self.gen_logvar.parameters()) + list(self.gen_mean.parameters()) \

        if self.learn_scale:
            self.iter_vae.append(self.scale)
        self.iter_alpha = list(self.rnn_alpha.parameters()) + list(self.mlp_alpha.parameters())
        self.iter_vae_kf = list(self.iter_kf) + list(self.iter_vae)
        self.iter_all = list(self.iter_vae_kf) + list(self.iter_alpha)

    def concat_iter(self, *iter_list):

        for i in iter_list:
            yield from i

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def inference(self, x):

        x_a = self.mlp_x_a(x)
        a_mean = self.inf_mean(x_a)
        a_logvar = self.inf_logvar(x_a)
        a_logvar = torch.log(torch.ones_like(a_logvar))
        a = self.reparameterization(a_mean, a_logvar)

        return a, a_mean, a_logvar

    def generation_x(self, a):
        a_x = self.mlp_a_x(a)

        # Feature generation
        # y_mean = self.gen_mean(a_x)
        # y_logvar = self.gen_logvar(a_x)
        # y_logvar = torch.log(1.0 * torch.ones_like(y_logvar))

        # y = torch.exp(y_logvar)
        # y = y_mean
        # y = self.reparameterization(y_mean, y_logvar)
        y_mean = None
        y_logvar = None
        y = None

        # Mask generation
        m_mean = self.gen_m_mean(a_x)

        return y, y_mean, y_logvar, m_mean

    def kf_smoother(self, a, u, K, A, B, C, R, Q, optimal_gain=False,
                    alpha_sq=1, generation=False):
        """"
        Kalman Smoother, refer to Murphy's book (MLAPP), section 18.3
        Difference from KVAE source code:
            - no imputation
            - only RNN for the calculation of alpha
            - different notations (rather than using same notations as Murphy's book ,we use notation from model KVAE)
            >>>> z_t = A_t * z_tm1 + B_t * u_t
            >>>> a_t = C_t * z_t
        Input:
            - a, (seq_len, bs, a_dim)
            - u, (seq_len, bs, u_dim)
            - alpha, (seq_len, bs, alpha_dim)
            - K, real number
            - A, (K, z_dim, z_dim)
            - B, (K, z_dim, u_dim)
            - C, (K, a_dim, z_dim)
            - R, (z_dim, z_dim)
            - Q , (a_dim, a_dim)
        """
        # Initialization
        seq_len = a.shape[1]
        if generation:
            seq_len += 1
        batch_size = a.shape[0]
        self.mu = torch.zeros((batch_size, self.z_dim)).to(
            self.device)  # (bs, z_dim), z_0
        self.Sigma = self.init_cov * torch.eye(self.z_dim).unsqueeze(0).repeat(
            batch_size, 1, 1).to(self.device)  # (bs, z_dim, z_dim), Sigma_0
        mu_pred = torch.zeros((seq_len, batch_size, self.z_dim)).to(
            self.device)  # (seq_len, bs, z_dim)
        mu_filter = torch.zeros((seq_len, batch_size, self.z_dim)).to(
            self.device)  # (seq_len, bs, z_dim)
        mu_smooth = torch.zeros((seq_len, batch_size, self.z_dim)).to(
            self.device)  # (seq_len, bs, z_dim)
        Sigma_pred = torch.zeros(
            (seq_len, batch_size, self.z_dim, self.z_dim)).to(
            self.device)  # (seq_len, bs, z_dim, z_dim)
        Sigma_filter = torch.zeros(
            (seq_len, batch_size, self.z_dim, self.z_dim)).to(
            self.device)  # (seq_len, bs, z_dim, z_dim)
        Sigma_smooth = torch.zeros(
            (seq_len, batch_size, self.z_dim, self.z_dim)).to(
            self.device)  # (seq_len, bs, z_dim, z_dim)

        # Calculate alpha, initial observation a_init is assumed to be zero and can be learned
        a_init_expand = self.a_init.unsqueeze(1).repeat(batch_size, 1, 1)  # (bs, 1, a_dim)
        if generation:
            a_tm1 = torch.cat([a_init_expand, a], 1)  # (bs, seq_len, a_dim)
        else:
            a_tm1 = torch.cat([a_init_expand, a[:, :-1, :]], 1)  # (bs, seq_len, a_dim)
        alpha = self.get_alpha(a_tm1)  # (bs, seq_len, K)

        # Calculate the mixture of A, B and C
        A_flatten = A.view(K, self.z_dim * self.z_dim)  # (K, z_dim*z_dim)
        B_flatten = B.view(K, self.z_dim * self.u_dim)  # (K, z_dim*u_dim)
        C_flatten = C.view(K, self.a_dim * self.z_dim)  # (K, a_dim*z_dim)
        A_mix = alpha.matmul(A_flatten).permute(1, 0, 2).view(seq_len, batch_size, self.z_dim,
                                             self.z_dim)
        B_mix = alpha.matmul(B_flatten).permute(1, 0, 2).view(seq_len, batch_size, self.z_dim,
                                             self.u_dim)
        C_mix = alpha.matmul(C_flatten).permute(1, 0, 2).view(seq_len, batch_size, self.a_dim,
                                             self.z_dim)

        # Forward filter
        for t in range(seq_len):

            # Mixture of A, B and C
            A_t = A_mix[t]  # (bs, z_dim, z_dim)
            B_t = B_mix[t]  # (bs, z_dim, u_dim)
            C_t = C_mix[t]  # (bs, a_dim, z_dim)

            ### Prediction step ###
            if t == 0:
                mu_t_pred = self.mu.unsqueeze(-1)  # (bs, z_dim, 1)
                Sigma_t_pred = self.Sigma
            else:
                u_t = u[:, t, :]  # (bs, u_dim)
                mu_t_pred = A_t.bmm(mu_t) + B_t.bmm(
                    u_t.unsqueeze(-1))  # (bs, z_dim, 1), z_{t|t-1}
                Sigma_t_pred = alpha_sq * A_t.bmm(Sigma_t).bmm(A_t.transpose(1,
                                                                             2)) + self.Q  # (bs, z_dim, z_dim), Sigma_{t|t-1}
                # alpha_sq (>=1) is fading memory control, which indicates how much you want to forgert past measurements, see more infos in 'FilterPy' library

            ### Measurement step ###
            # Residual
            a_pred = C_t.bmm(mu_t_pred)  # (bs, a_dim, z_dim) x (bs, z_dim, 1)
            res_t = a[:, t, :].unsqueeze(-1) - a_pred  # (bs, a_dim, 1)

            # Kalman gain
            S_t = C_t.bmm(Sigma_t_pred).bmm(
                C_t.transpose(1, 2)) + self.R  # (bs, a_dim, a_dim)
            S_t_inv = S_t.inverse() # TODO: does this inversion take long?
            K_t = Sigma_t_pred.bmm(C_t.transpose(1, 2)).bmm(
                S_t_inv)  # (bs, z_dim, a_dim)

            # Update
            mu_t = mu_t_pred + K_t.bmm(res_t)  # (bs, z_dim, 1)
            I_KC = self._I - K_t.bmm(C_t)  # (bs, z_dim, z_dim)
            if optimal_gain:
                Sigma_t = I_KC.bmm(
                    Sigma_t_pred)  # (bs, z_dim, z_dim), only valid with optimal Kalman gain
            else:
                Sigma_t = I_KC.bmm(Sigma_t_pred).bmm(
                    I_KC.transpose(1, 2)) + K_t.matmul(self.R).matmul(
                    K_t.transpose(1, 2))  # (bs, z_dim, z_dim), general case

            # Save cache
            mu_pred[t] = mu_t_pred.view(batch_size, self.z_dim)
            ### THIS WAS MISSING ####
            mu_filter[t] = mu_t.view(batch_size, self.z_dim)
            #########################
            Sigma_pred[t] = Sigma_t_pred
            Sigma_filter[t] = Sigma_t

        # Add the final state from filter to the smoother as initialization
        mu_smooth[-1] = mu_filter[-1]
        Sigma_smooth[-1] = Sigma_filter[-1]

        # Backward smooth, reverse loop from pernultimate state
        for t in range(seq_len - 2, -1, -1):
            # Backward Kalman gain
            J_t = Sigma_filter[t].bmm(A_mix[t + 1].transpose(1, 2)).bmm(
                Sigma_pred[t + 1].inverse())  # (bs, z_dim, z_dim)

            # Backward smoothing
            dif_mu_tp1 = (mu_smooth[t + 1] - mu_pred[t + 1]).unsqueeze(
                -1)  # (bs, z_dim, 1)
            mu_smooth[t] = mu_filter[t] + J_t.matmul(dif_mu_tp1).view(
                batch_size, self.z_dim)  # (bs, z_dim)
            dif_Sigma_tp1 = Sigma_smooth[t + 1] - Sigma_pred[
                t + 1]  # (bs, z_dim, z_dim)
            Sigma_smooth[t] = Sigma_filter[t] + J_t.bmm(dif_Sigma_tp1).bmm(
                J_t.transpose(1, 2))  # (bs, z_dim, z_dim)

        # Transpose all returns to have batch_size as leading dimension
        A_mix = A_mix.permute(1, 0, 2, 3)
        B_mix = B_mix.permute(1, 0, 2, 3)
        C_mix = C_mix.permute(1, 0, 2, 3)

        mu_smooth = mu_smooth.permute(1, 0, 2)
        Sigma_smooth = Sigma_smooth.permute(1, 0, 2, 3)

        # Generate a from smoothed z
        a_gen = C_mix.matmul(mu_smooth.unsqueeze(-1)).view(batch_size, seq_len,
                                                           self.a_dim)  # (bs, seq_len, a_dim)

        return a_gen, mu_smooth, Sigma_smooth, A_mix, B_mix, C_mix

    def get_alpha(self, a_tm1):
        """
        Dynamics parameter network alpha for mixing transitions in a SSM
        Unlike original code, we only propose RNN here
        """

        alpha, _ = self.rnn_alpha(a_tm1)  # (bs, seq_len, dim_alpha)
        alpha = self.mlp_alpha(
            alpha)  # (bs, seq_len, K), softmax on K dimension

        return alpha

    def forward_vae(self, x, compute_loss=False):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(-1, 0, 1)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        a, a_mean, a_logvar = self.inference(x)
        y = self.generation_x(a)

        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss_vae(x, y, a_mean,
                                                               a_logvar,
                                                               batch_size,
                                                               seq_len)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)

        self.y = y.permute(1, -1, 0).squeeze()

        return self.y

    def forward(self, x, u, compute_loss=False):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(u.shape) == 2:
            u = u.unsqueeze(1)
        # x = x.permute(-1, 0, 1)
        x = x.permute(0, -1, 1)
        u = u.permute(0, -1, 1)

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # main part
        a, a_mean, a_logvar = self.inference(x)
        # batch_size = a.shape[1]
        u_0 = torch.zeros(batch_size, 1, self.a_dim).to(self.device)
        u_cat = torch.cat((u_0, a[:,:-1,:]), 1)
        u_cat = torch.cat((u_cat, u), 2)
        a_gen, mu_smooth, Sigma_smooth, A_mix, B_mix, C_mix = self.kf_smoother(
            a, u_cat, self.K, self.A, self.B, self.C, self.R, self.Q)
        if not self.use_smoothed_a:
            a_gen = a
        y, y_mean, y_logvar, m_mean = self.generation_x(a_gen)

        # calculate loss
        if compute_loss:
            loss_tot, loss_vae, loss_lgssm, debug_losses = self.get_loss(x, y, y_mean, y_logvar, m_mean, u_cat,
                                                           a, a_mean, a_logvar,
                                                           mu_smooth,
                                                           Sigma_smooth,
                                                           A_mix, B_mix, C_mix,
                                                           self.scale_reconstruction,
                                                           seq_len, batch_size)
            self.loss = (loss_tot, loss_vae, loss_lgssm, debug_losses)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)

        # self.y = y.permute(0, -1, 1).squeeze()
        self.y = y


        return self.y, m_mean

    def get_loss_vae(self, x, y, a_mean, a_logvar, batch_size, seq_len, beta=1):

        loss_recon = torch.sum(x / y - torch.log(x / y) - 1)
        loss_KLD = -0.5 * torch.sum(a_logvar - a_logvar.exp() - a_mean.pow(2))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD

    def get_loss(self, x, y, y_mean, y_logvar, m_mean, u, a, a_mean, a_logvar, mu_smooth, Sigma_smooth,
                 A, B, C, scale_reconstruction=1, seq_len=150, batch_size=32):
        # Split mask from original input
        m_idxs = np.arange(self.m_dim)
        # feats_idxs = np.ones(x.shape[-1], np.bool)
        # feats_idxs[m_idxs] = 0
        # x_feats = x[..., feats_idxs]
        x_m = x[..., m_idxs]

        # log p_{\theta}(x | a_hat), Gaussian
        # log_px_given_a_feats = - torch.log(y) - x / y
        # log_px_given_a_feats = - torch.square(x - y)
        # Gaussian log-likelihood
        # log_px_given_a_feats = - 0.5 * y_logvar - torch.pow(x_feats - y_mean, 2) / (
        #             2 * torch.exp(y_logvar))
        # log_px_given_a_feats = - torch.pow(x_feats - y_mean, 2) / (2 * torch.exp(y_logvar))
        # Bernouli log-likelihood
        eps = 1e-6
        m_mean = torch.clamp(m_mean, eps, 1-eps)
        log_px_given_a_masks = x_m * torch.log(m_mean) + (1 - x_m) * torch.log(1 - m_mean)
        if self.sample_m:
            # Sample from distribution
            bn_mask = Bernoulli(probs=m_mean)
            missing_mask = bn_mask.sample().int()
        else:
            # Threshold probabilities
            missing_mask = torch.round(m_mean).int()

        # Construct mask for reconstruction loss of features
        # if self.x_dim > 2*self.m_dim:
        #     missing_mask = torch.tile(missing_mask, (1,1,2))
        # # Add row of ones to mask to keep information about labels at each step
        # label_ones = torch.ones(missing_mask.shape[0], missing_mask.shape[1], 1).to(self.device)
        # missing_mask = torch.cat((missing_mask, label_ones), dim=2)

        # Mask feature loss
        # mask_zeros = torch.zeros_like(log_px_given_a_feats).to(self.device)
        # log_px_given_a_feats = torch.where(missing_mask.bool(),
        #                                    log_px_given_a_feats,
        #                                    mask_zeros)

        # log q_{\phi}(a_hat | x), Gaussian
        log_qa_given_x = - 0.5 * a_logvar - torch.pow(a - a_mean, 2) / (
                    2 * torch.exp(a_logvar))
        # log_qa_given_x = - torch.pow(a - a_mean, 2) / (2 * torch.exp(a_logvar))

        # log p_{\gamma}(a_tilde, z_tilde | u) < in sub-comment, 'tilde' is hidden for simplification >
        # >>> log p(z_t | z_tm1, u_t), transition
        mvn_smooth = MultivariateNormal(mu_smooth, Sigma_smooth)
        z_smooth = mvn_smooth.sample()  # # (bs, seq_len, z_dim)
        Az_tm1 = A[:,:-1,:].matmul(z_smooth[:,:-1,:].unsqueeze(-1)).view(batch_size,
                                                                         seq_len - 1,
                                                                         -1)  # (seq_len, bs, z_dim)
        Bu_t = B[:,:-1,:].matmul(u[:,1:,:].unsqueeze(-1)).view(batch_size, seq_len - 1,
                                                        -1)  # (seq_len, bs, z_dim)
        mu_t_transition = Az_tm1 + Bu_t
        z_t_transition = z_smooth[:,1:,:]
        mvn_transition = MultivariateNormal(mu_t_transition, self.Q)
        log_prob_transition = mvn_transition.log_prob(z_t_transition)
        # >>> log p(z_0 | z_init), init state
        z_0 = z_smooth[:,0,:]
        mvn_0 = MultivariateNormal(self.mu, self.Sigma)
        log_prob_0 = mvn_0.log_prob(z_0)
        # >>> log p(a_t | z_t), emission
        Cz_t = C.matmul(z_smooth.unsqueeze(-1)).view(batch_size, seq_len,
                                                     self.a_dim)
        mvn_emission = MultivariateNormal(Cz_t, self.R)
        log_prob_emission = mvn_emission.log_prob(a)
        # >>> log p_{\gamma}(a_tilde, z_tilde | u)
        log_paz_given_u = torch.cat(
            [log_prob_transition, log_prob_0.unsqueeze(1)],
            1) + log_prob_emission

        # log p_{\gamma}(z_tilde | a_tilde, u)
        # >>> log p(z_t | a, u)
        log_pz_given_au = mvn_smooth.log_prob(z_smooth)

        # Normalization
        # log_px_given_a_feats = torch.sum(log_px_given_a_feats) / (batch_size * seq_len)
        log_px_given_a_masks = torch.sum(log_px_given_a_masks) / (batch_size * seq_len)
        # log_px_given_a = torch.exp(self.scale) * log_px_given_a_feats + log_px_given_a_masks
        log_px_given_a = log_px_given_a_masks
        log_qa_given_x = torch.sum(log_qa_given_x) / (batch_size * seq_len)
        log_paz_given_u = torch.sum(log_paz_given_u) / (batch_size * seq_len)
        log_pz_given_au = torch.sum(log_pz_given_au) / (batch_size * seq_len)

        # Loss
        loss_vae = - scale_reconstruction * log_px_given_a + log_qa_given_x
        loss_lgssm = - log_paz_given_u + log_pz_given_au
        loss_tot = loss_vae + loss_lgssm

        # Individual loss terms for debugging
        loss_x_a = - scale_reconstruction * log_px_given_a
        loss_a_x = log_qa_given_x
        loss_az_u = - log_paz_given_u
        loss_z_au = log_pz_given_au

        return loss_tot, loss_vae, loss_lgssm, (loss_x_a, loss_a_x, loss_az_u, loss_z_au)

    def generate_synth(self, labels, seq_len, det_gen=False, batch_size=64):
        """
        Conditionally generates new synthetic data by sampling from the latent space.

        Args:
            labels: labels to condition on during generation, array: [N,]
            seq_len: length of sequence to generate, int
            det_gen: whether or not to add stochastic noise to the generation, bool
            batch_size: batch size for mini batch data generation, int
        """
        # Flattened system matrices
        A_flatten = self.A.view(self.K, self.z_dim * self.z_dim)  # (K, z_dim*z_dim)
        B_flatten = self.B.view(self.K, self.z_dim * self.u_dim)  # (K, z_dim*u_dim)
        C_flatten = self.C.view(self.K, self.a_dim * self.z_dim)  # (K, a_dim*z_dim)

        # Make labels dataloader
        labels_loader = DataLoader(torch.Tensor(labels), batch_size=batch_size)

        # Outer batch loop
        x_batched = []
        for label_batch in labels_loader:
            a_seq = []
            batch_len = len(label_batch)
            label_batch = label_batch.to(self.device)
            label_batch = label_batch.view(batch_len, self.u_dim - self.a_dim)


            # Sample from prior
            mu = torch.zeros((batch_len, self.z_dim)).to(self.device)
            Sigma = self.init_cov * torch.eye(self.z_dim).unsqueeze(0).repeat(
                batch_len, 1, 1).to(self.device)
            mvn_0 = MultivariateNormal(mu, Sigma)
            z_t = mvn_0.sample().unsqueeze(-1)

            # Get initial a
            a_init = self.a_init.unsqueeze(1).repeat(batch_len, 1, 1)

            # Initial u includes a_init
            u_t = torch.cat((a_init.permute(0,2,1), label_batch.unsqueeze(-1)), 1)

            # Initial dynamics mixture
            alpha, (h, c) = self.rnn_alpha(a_init)
            alpha = self.mlp_alpha(alpha)

            # Stochastic noise
            if det_gen:
                epsilon = torch.zeros((batch_len, seq_len, self.z_dim))
                delta = torch.zeros((batch_len, seq_len, self.a_dim))
            else:
                # Pre-compute noise at each step
                noise_trans = MultivariateNormal(torch.zeros(self.z_dim).to(self.device), self.Q)
                epsilon = noise_trans.sample((batch_len, seq_len))
                noise_emis = MultivariateNormal(torch.zeros(self.a_dim).to(self.device), self.R)
                delta = noise_emis.sample((batch_len, seq_len))
            epsilon = epsilon.to(self.device)
            delta = delta.to(self.device)

            for t in range(seq_len):
                C_t = alpha.matmul(C_flatten).view(batch_len, self.a_dim, self.z_dim)
                # Output for current time step
                a_t = C_t.matmul(z_t) + delta[:, t, :].unsqueeze(-1)

                # Save output at step t
                a_seq.append(a_t)

                # Dynamics at current time step
                alpha, (h, c) = self.rnn_alpha(a_t.permute(0,2,1), (h, c))
                alpha = self.mlp_alpha(alpha)

                A_t = alpha.matmul(A_flatten).view(batch_len, self.z_dim, self.z_dim)
                B_t = alpha.matmul(B_flatten).view(batch_len, self.z_dim, self.u_dim)

                # Compute next step of z
                z_t = A_t.matmul(z_t) + B_t.matmul(u_t)\
                      + epsilon[:, t, :].unsqueeze(-1)

                # Get next input
                u_t = torch.cat((a_t, label_batch.unsqueeze(-1)), 1)

            x_seq, _, _, m_mean = self.generation_x(torch.stack(a_seq, dim=1).squeeze())
            eps = 1e-6
            m_mean = torch.clamp(m_mean, eps, 1 - eps)
            if self.sample_m:
                # Sample from distribution
                bn_mask = Bernoulli(probs=m_mean)
                missing_mask = bn_mask.sample().int()
            else:
                # Threshold probabilities
                missing_mask = torch.round(m_mean).int()

            # Concatenate masks back with features
            # x_m_seq = torch.cat((x_seq[..., :self.m_dim], missing_mask, x_seq[..., self.m_dim:]), dim=2)

            # x_batched.append(x_m_seq)
            x_batched.append(missing_mask)

        x_cat = torch.cat(x_batched)

        # Split features from appended labels
        # X_synth = x_cat[:, :, :-1]
        X_synth = x_cat
        X_synth = X_synth.permute(0,2,1)

        return X_synth.cpu().detach().numpy(), labels


    def get_info(self):

        info = []
        info.append("----- VAE -----")
        for layer in self.mlp_x_a:
            info.append(str(layer))
        info.append(self.inf_mean)
        info.append(self.inf_logvar)
        for layer in self.mlp_a_x:
            info.append(str(layer))
        info.append(self.gen_logvar)

        info.append("----- Dynamics -----")
        info.append(self.rnn_alpha)
        info.append(self.mlp_alpha)

        info.append("----- LGSSM -----")
        info.append("A dimension: {}".format(str(self.A.shape)))
        info.append("B dimension: {}".format(str(self.B.shape)))
        info.append("C dimension: {}".format(str(self.C.shape)))
        info.append("transition noise level: {}".format(self.noise_transition))
        info.append("emission noise level: {}".format(self.noise_emission))
        info.append("scale for initial B and C: {}".format(self.init_kf_mat))
        info.append("scale for initial covariance: {}".format(self.init_cov))

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        info.append(F'Total nr. of trainable params: {params}')

        return info


if __name__ == '__main__':
    x_dim = 257
    device = 'cpu'

    kvae = KVAE_miss(x_dim).to(device)

    # x = torch.rand([2,257,3])
    # y, loss, _, _ = kvae.forward(x)
    # print(loss)
