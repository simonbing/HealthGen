"""
2021 Simon Bing, ETHZ, MPI IS
"""
import os
import numpy as np
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import wandb
import logging
from healthgen.generation.base_gen_model import BaseGenModel
from healthgen.generation.models import KVAE
from absl import flags, app

FLAGS=flags.FLAGS
# Model
flags.DEFINE_integer('a_dim', 32, 'Feature extraction dimension.')
flags.DEFINE_list('dense_x_a', [256,128], 'List of dimensions for dense x_a layers.')
flags.DEFINE_list('dense_a_x', [128,256], 'List of dimensions for dense a_x layers.')
flags.DEFINE_float('init_kf_mat', 0.05, 'Kf init mat') # TODO: help
flags.DEFINE_float('noise_transition', 0.08, 'Noise for Kalman transition.')
flags.DEFINE_float('noise_emission', 0.03, 'Noise for Kalman emission.')
flags.DEFINE_float('init_cov', 20.0, 'Initial KVAE covariance.')
flags.DEFINE_integer('K', 10, 'K') # TODO: help
flags.DEFINE_integer('dim_rnn_alpha', 50, 'Hidden dimension for alpha RNN.')
flags.DEFINE_integer('num_rnn_alpha', 1, 'Number of layers for alpha RNN.')
flags.DEFINE_float('scale_recon', 1.0, 'Reconstruction scale KVAE.') # TODO: help
flags.DEFINE_bool('use_smoothed_a', False, 'Whether to use the a seq. from the smoothed posterior or seq.'
                                           'obtained directly from the encoder.')
# Training
flags.DEFINE_bool('scheduler_training', False, 'Train with scheduler or not.')
flags.DEFINE_integer('only_vae_epochs', 10, 'Epochs in which to only train VAE.')
flags.DEFINE_integer('kf_update_epochs', 10, 'Epochs in which to only train KF.')

class KVAEGenModel(BaseGenModel):
    def __init__(self, seed, x_dim, u_dim, a_dim, z_dim, activation, dropout, dense_x_a,
                 dense_a_x, init_kf_mat, noise_transition, noise_emission, init_cov,
                 K, dim_rnn_alpha, num_rnn_alpha, scale_recon, use_smoothed_a):
        # Model parameters
        # General
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.activation = activation
        self.dropout = dropout
        self.scale_recon = scale_recon
        # VAE
        self.dense_x_a = [int(size) for size in dense_x_a]
        self.dense_a_x = [int(size) for size in dense_a_x]
        self.use_smoothed_a = use_smoothed_a
        # LGSSM
        self.init_kf_mat = init_kf_mat
        self.noise_transition = noise_transition
        self.noise_emission = noise_emission
        self.init_cov = init_cov
        # Dynamics params (alpha)
        self.K = K
        self.dim_rnn_alpha = dim_rnn_alpha
        self.num_rnn_alpha = num_rnn_alpha

        super().__init__(seed)

    def build_model(self):
        model  = KVAE(x_dim=self.x_dim, u_dim=self.u_dim, a_dim=self.a_dim, z_dim=self.z_dim,
                      activation=self.activation, dense_x_a=self.dense_x_a,
                      dense_a_x=self.dense_a_x,init_kf_mat=self.init_kf_mat,
                      noise_transition=self.noise_transition,
                      noise_emission=self.noise_emission, init_cov=self.init_cov,
                      K=self.K, dim_RNN_alpha=self.dim_rnn_alpha,
                      num_RNN_alpha=self.num_rnn_alpha, dropout_p=self.dropout,
                      scale_reconstruction=self.scale_recon, device=self.device,
                      use_smoothed_a=self.use_smoothed_a).to(self.device)
        return model

    def build_dataloader(self, X, y, batch_size):
        if FLAGS.data == 'mimic':
            if FLAGS.data_mode == 'feats':
                X_train = X['X_train']
                X_val = X['X_val']
            elif FLAGS.data_mode == 'mask':
                X_train = X['m_train']
                X_val = X['m_val']
            elif FLAGS.data_mode == 'feats_mask':
                X_train = np.concatenate((X['X_train'], X['m_train']), axis=1)
                X_val = np.concatenate((X['X_val'], X['m_val']), axis=1)
            elif FLAGS.data_mode == 'all':
                X_train = np.concatenate((X['X_train'], X['m_train'], X['delta_t_train']), axis=1)
                X_val = np.concatenate((X['X_val'], X['m_val'], X['delta_t_val']), axis=1)
            # Expand labels to all time steps
            # time_steps = X_train.shape[2]
            # y_train = np.tile(y['y_train'], (time_steps, 1)).transpose()
            # y_val = np.tile(y['y_val'], (time_steps, 1)).transpose()

            # Add labels as input feature
            # data_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
            # data_val = np.concatenate((X_val, np.expand_dims(y_val, axis=1)), axis=1)
            # Condition on labels using u_t
            data_train = X_train
            data_val = X_val
        elif FLAGS.data == 'ball_box':
            X_train = np.stack((X['x_pos_train'], X['y_pos_train']), axis=1)
            X_val = np.stack((X['x_pos_val'], X['y_pos_val']), axis=1)

            data_train = X_train
            data_val = X_val
        else:
            raise TypeError

        # Expand labels to all time steps
        time_steps = X_train.shape[2]
        y_train = np.tile(y['y_train'], (time_steps, 1)).transpose()
        y_val = np.tile(y['y_val'], (time_steps, 1)).transpose()

        # Check if x_dim of model fits data
        assert self.x_dim == data_train.shape[1], F'x_dim specified in model does not match data!' \
                                                  F'Should be {data_train.shape[1]}!'

        train_dataset = TensorDataset(torch.Tensor(data_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(data_val), torch.Tensor(y_val))

        dataloader_train = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=12, pin_memory=self.use_cuda)
        dataloader_val = DataLoader(val_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=12, pin_memory=self.use_cuda)

        return dataloader_train, dataloader_val

    def init_optimizer(self):
        self.optimizer_vae = torch.optim.Adam(self.model.iter_vae, lr=FLAGS.gen_lr)
        self.optimizer_vae_kf = torch.optim.Adam(self.model.iter_vae_kf, lr=FLAGS.gen_lr)
        self.optimizer_all = torch.optim.Adam(self.model.iter_all, lr=FLAGS.gen_lr)

    def train_loop(self, train_dataloader, val_dataloader):
        """
        Special training loop for KVAE model.
        """
        logging.info('Training of KVAE started!')
        # Training preparation
        train_out_path = os.path.join(self.gen_out_path, 'training')
        if not os.path.isdir(train_out_path):
            os.makedirs(train_out_path)

        train_num = train_dataloader.sampler.num_samples
        val_num = val_dataloader.sampler.num_samples

        learning_rate_decay_step = FLAGS.gen_lr_decay_step

        best_val_loss = np.inf
        best_epoch = 0
        stop_patience = 0

        self.model.train()
        torch.autograd.set_detect_anomaly(bool(~self.use_cuda))

        # Train with mini-batch SGD
        for epoch in range(1, FLAGS.gen_epochs+1):
            if FLAGS.scheduler_training:
                if epoch < FLAGS.only_vae_epochs:
                    optimizer = self.optimizer_vae
                    logging.info(F'Epoch: {epoch}, training only VAE of KVAE!')
                elif epoch < FLAGS.only_vae_epochs + FLAGS.kf_update_epochs:
                    optimizer = self.optimizer_vae_kf
                    logging.info(F'Epoch: {epoch}, training VAE + KF of KVAE!')
                else:
                    optimizer = self.optimizer_all
                    logging.info(F'Epoch: {epoch}, training all variables of KVAE!')
            else:
                optimizer = self.optimizer_all

            if learning_rate_decay_step != 0:
                # Learning rate scheduler
                milestones = np.arange(learning_rate_decay_step, FLAGS.gen_epochs, learning_rate_decay_step)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.85)

            # Batch training
            train_loss_tot = 0
            train_loss_vae = 0
            train_loss_lgssm = 0

            for train_batch, train_label in train_dataloader:
                train_batch = train_batch.to(self.device)
                train_label = train_label.to(self.device)
                # Zero gradients
                for param in self.model.parameters():
                    param.grad = None
                # Forward pass
                recon_train_batch = self.model(train_batch, train_label, compute_loss=True)
                # Compute loss
                loss_tot, loss_vae, loss_lgssm = self.model.loss
                # Backward pass
                loss_tot.backward()
                # Update weights
                optimizer.step()

                train_loss_tot += loss_tot.item()
                train_loss_vae += loss_vae.item()
                train_loss_lgssm += loss_lgssm.item()

            # Log (normalized) training losses
            wandb.log({'train_loss_tot': train_loss_tot / train_num,
                       'train_loss_vae': train_loss_vae / train_num,
                       'train_loss_lgssm': train_loss_lgssm / train_num,
                       'epoch': epoch})

            if epoch % FLAGS.gen_val_freq == 0:
                val_loss_tot = 0
                val_loss_vae = 0
                val_loss_lgssm = 0
                with torch.no_grad():
                    for val_batch, val_label in val_dataloader:
                        val_batch = val_batch.to(self.device)
                        val_label = val_label.to(self.device)
                        # Forward pass
                        recon_val_batch = self.model(val_batch, val_label, compute_loss=True)
                        # Compute loss
                        val_loss_tot, loss_vae, loss_lgssm = self.model.loss

                        val_loss_tot += loss_tot.item()
                        val_loss_vae += loss_vae.item()
                        val_loss_lgssm += loss_lgssm.item()

                    # Log (normalized) validation losses
                    wandb.log({'val_loss_tot': val_loss_tot / val_num,
                               'val_loss_vae': val_loss_vae / val_num,
                               'val_loss_lgssm': val_loss_lgssm / val_num,
                               'epoch': epoch})

                # Intermediate model saving
                model_path = os.path.join(train_out_path,F'{FLAGS.gen_model}_epoch_{epoch}_model.pt')
                logging.info(F'Saved model at epoch {epoch}!')
                torch.save(self.model.state_dict(), model_path)

                # Early stopping
                if val_loss_tot < best_val_loss:
                    best_model_state_dict = deepcopy(self.model.state_dict())
                    stop_patience = 0
                    best_val_loss = val_loss_tot
                    best_epoch = epoch
                else:
                    stop_patience += 1

                # Stop training if early stopping triggers
                if stop_patience == FLAGS.gen_early_stop_patience:
                    logging.info(
                        F'Early stopping patience achieved. Stopping training at epoch {epoch}!')
                    break

            if learning_rate_decay_step != 0:
                scheduler.step(epoch=epoch)

        # Save the best model
        best_model_path = os.path.join(train_out_path, F'{FLAGS.gen_model}_best_model.pt')
        logging.info(F'Saved best model from epoch {best_epoch}!')
        torch.save(best_model_state_dict, best_model_path)
        logging.info(
            F'Training of KVAE successfully finished!')

    def reconstruct(self, X, y, N):
        """
        Reconstructs input data.

        Args:
            X, y: dicts of input features and labels
            N: Number of samples to reconstruct

        Returns:
            data_orig: array containing original input data (features and labels)
            data_recon: array containing reconstructed data (features and labels)
        """
        _, val_dataloader = self.build_dataloader(X, y, N, )
        # Get single batch of size N from loader
        data_orig, label_orig = next(iter(val_dataloader))
        data_orig = data_orig.to(self.device)
        label_orig = label_orig.to(self.device)

        with torch.no_grad():
            data_recon = self.model(data_orig, label_orig, compute_loss=False).cpu().detach().numpy()

        return data_orig.cpu().detach().numpy(), data_recon