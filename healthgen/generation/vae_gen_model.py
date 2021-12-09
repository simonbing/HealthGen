"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from healthgen.generation.base_gen_model import BaseGenModel
from healthgen.generation.models import VAE
from absl import flags, app

FLAGS=flags.FLAGS
# Model
flags.DEFINE_enum('encoder', 'mlp', ['mlp', 'conv'], 'Which encoder/decoder architecture to use.')
flags.DEFINE_list('dense_x_z', [256,128], 'List of dimensions for dense x_h layers.')
flags.DEFINE_list('dense_z_x', [128,256], 'List of dimensions for hz_x layers.')
flags.DEFINE_list('conv_x_z', [128,64], 'List of dimensions for encoder conv layers.')
flags.DEFINE_list('conv_z_x', [64,128], 'List of dimensions for decoder conv layers.')


class VAEGenModel(BaseGenModel):
    def __init__(self, seed, x_dim, z_dim, seq_len, activation, dropout, encoder, dense_x_z,
                 dense_z_x, conv_x_z, conv_z_x, beta, data_mode, mask_loss):
        # Model parameters
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.activation = activation
        self.dropout_p = dropout
        self.encoder = encoder
        self.data_mode = data_mode
        # Inference
        self.dense_x_z = [int(i) for i in dense_x_z]
        self.conv_x_z = [int(i) for i in conv_x_z]
        # Generation
        self.dense_z_x = [int(i) for i in dense_z_x]
        self.conv_z_x = [int(i) for i in conv_z_x]
        # Beta
        self.beta = beta
        # Training
        self.mask_loss = mask_loss

        super().__init__(seed)

    def build_model(self):
        model = VAE(x_dim=self.x_dim, z_dim=self.z_dim, seq_len=self.seq_len, activation=self.activation,
                    dropout_p=self.dropout_p, encoder=self.encoder, dense_x_z=self.dense_x_z,
                    dense_z_x=self.dense_z_x, conv_x_z=self.conv_x_z, conv_z_x=self.conv_z_x,
                    beta=self.beta, data_mode=self.data_mode, mask_loss=self.mask_loss, device=self.device).to(self.device)

        return model

    def build_dataloader(self, X, y, batch_size):
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
        time_steps = X_train.shape[2]
        y_train = np.expand_dims(np.tile(y['y_train'], (time_steps, 1)).transpose(), axis=1)
        y_val = np.expand_dims(np.tile(y['y_val'], (time_steps, 1)).transpose(), axis=1)

        # Check if x_dim of model fits data
        assert self.x_dim == X_train.shape[1], F'x_dim specified in model does not match data!' \
                F'Should be {X_train.shape[1]}!'

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

        dataloader_train = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=6)
        dataloader_val = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=6)

        return dataloader_train, dataloader_val