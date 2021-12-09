"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from healthgen.generation.base_gen_model import BaseGenModel
from healthgen.generation.models import HealthGen
from absl import flags

FLAGS=flags.FLAGS
# Model
flags.DEFINE_integer('v_dim', 16, 'Dimension for static latent variable.')
flags.DEFINE_list('dense_x_v', [256,128], 'List of dimensions for x_v layers.')
flags.DEFINE_list('dense_v_m', [256,128], 'List of dimensions for v_m layers.')

class HealthGenModel(BaseGenModel):
    def __init__(self, seed, x_dim, y_dim, v_dim, z_dim, seq_len, activation, dropout,
                 dense_x_v, dense_x_h, dense_hx_g, dense_gz_z, dim_RNN_h, num_RNN_h, dim_RNN_g,
                 num_RNN_g, dense_hz_z, dense_hz_x, dense_v_m, beta):
        # Model parameters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.v_dim = v_dim
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.activation = activation
        self.dropout = dropout
        # Inference
        self.dense_x_v = [int(i) for i in dense_x_v]
        self.dense_x_h = [int(i) for i in dense_x_h]
        self.dense_hx_g = [int(i) for i in dense_hx_g]
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        self.dense_gz_z = [int(i) for i in dense_gz_z]

        # Generation
        self.dense_hz_z = [int(i) for i in dense_hz_z]
        self.dense_hz_x = [int(i) for i in dense_hz_x]
        self.dense_v_m = [int(i) for i in dense_v_m]

        # Training
        self.beta = beta

        super().__init__(seed)


    def build_model(self):
        model = HealthGen(x_dim=self.x_dim, y_dim=self.y_dim, v_dim=self.v_dim, z_dim=self.z_dim, seq_len=self.seq_len,
                          activation=self.activation, dropout_p=self.dropout, dense_x_v=self.dense_x_v,
                          dense_x_h=self.dense_x_h, dense_hx_g=self.dense_hx_g, dense_gz_z=self.dense_gz_z, dim_RNN_h=self.dim_RNN_h,
                          num_RNN_h=self.num_RNN_h, dim_RNN_g=self.dim_RNN_g, num_RNN_g=self.num_RNN_g,
                          dense_hz_z=self.dense_hz_z, dense_hz_x=self.dense_hz_x, dense_v_m=self.dense_v_m, beta=self.beta, device=self.device)

        return model


    def build_dataloader(self, X, y, batch_size):
        assert FLAGS.data_mode == 'feats_mask', 'data_mode must be feats_mask for this model!'

        # Pass features in some kind of discernable fashion, maybe stack in new dimension
        X_train = np.stack((X['X_train'], X['m_train']), axis=1)
        X_val = np.stack((X['X_val'], X['m_val']), axis=1)

        # Expand labels to all time steps
        time_steps = X_train.shape[3]
        y_train = np.expand_dims(np.tile(y['y_train'], (time_steps, 1)).transpose(), axis=1)
        y_val = np.expand_dims(np.tile(y['y_val'], (time_steps, 1)).transpose(), axis=1)
        # Concatenate static variables to labels, if necessary
        if FLAGS.cond_static:
            # Select static variables of interest
            if FLAGS.static_vars is None:
                static_vars_idxs = np.arange(len(y['feature_names']))
            else:
                static_vars_idxs = np.flatnonzero(np.in1d(y['feature_names'], FLAGS.static_vars))
            c_train = np.tile(y['c_train'][:,static_vars_idxs].transpose(), (time_steps, 1, 1)).transpose(2,1,0)
            c_val = np.tile(y['c_val'][:,static_vars_idxs].transpose(), (time_steps, 1, 1)).transpose(2,1,0)

            # Concatenate with labels
            y_train = np.concatenate((y_train, c_train), axis=1)
            y_val = np.concatenate((y_val, c_val), axis=1)

        # Check if x_dim of model fits data
        assert self.x_dim == X_train.shape[2], F'x_dim specified in model does not match data!' \
                F'Should be {X_train.shape[2]}!'

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

        dataloader_train = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=6)
        dataloader_val = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=6)

        return dataloader_train, dataloader_val