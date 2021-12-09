"""
2021 Simon Bing, ETHZ, MPI IS
"""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from healthgen.generation.base_gen_model import BaseGenModel
from healthgen.generation.models import SRNN
from absl import flags, app

FLAGS=flags.FLAGS
# Model
flags.DEFINE_list('dense_x_h', [], 'List of dimensions for dense x_h layers.')
flags.DEFINE_integer('dim_rnn_h', 128, 'Hidden dimension for h RNN.')
flags.DEFINE_integer('num_rnn_h', 1, 'Number of layers for h RNN.')
flags.DEFINE_list('dense_hx_g', [], 'List of dimensions for dense hx_g layers.')
flags.DEFINE_integer('dim_rnn_g', 128, 'Hidden dimension for g RNN.')
flags.DEFINE_integer('num_rnn_g', 1, 'Number of layers for g RNN.')
flags.DEFINE_list('dense_gz_z', [64,32], 'List of dimensions for gz_z layers.')
flags.DEFINE_list('dense_hz_z', [64,32], 'List of dimensions for hz_z layers.')
flags.DEFINE_list('dense_hz_x', [256], 'List of dimensions for hz_x layers.')

class SRNNGenModel(BaseGenModel):
    def __init__(self, seed, x_dim, z_dim, activation, dropout, dense_x_h, dim_rnn_h,
                 num_rnn_h, dense_hx_g, dim_rnn_g, num_rnn_g, dense_gz_z, dense_hz_z,
                 dense_hz_x, beta):
        # Model parameters
        # General
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.activation = activation
        self.dropout_p = dropout
        # Deterministic
        self.dense_x_h = dense_x_h
        self.dim_RNN_h = dim_rnn_h
        self.num_RNN_h = num_rnn_h
        # Inference
        self.dense_hx_g = dense_hx_g
        self.dim_RNN_g = dim_rnn_g
        self.num_RNN_g = num_rnn_g
        self.dense_gz_z = dense_gz_z
        # Prior
        self.dense_hz_z = dense_hz_z
        # Generation
        self.dense_hz_x = dense_hz_x
        # Beta-vae
        self.beta = beta

        super().__init__(seed)

    def build_model(self):
        model = SRNN(x_dim=self.x_dim, z_dim=self.z_dim, activation=self.activation,
                     dense_x_h=self.dense_x_h,
                     dim_RNN_h=self.dim_RNN_h, num_RNN_h=self.num_RNN_h,
                     dense_hx_g=self.dense_hx_g,
                     dim_RNN_g=self.dim_RNN_g, num_RNN_g=self.num_RNN_g,
                     dense_gz_z=self.dense_gz_z,
                     dense_hz_x=self.dense_hz_x,
                     dense_hz_z=self.dense_hz_z,
                     dropout_p=self.dropout_p, beta=self.beta, device=self.device).to(self.device)

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

        # Add labels as input feature
        data_train = np.concatenate((X_train, y_train), axis=1)
        data_val = np.concatenate((X_val, y_val), axis=1)

        # Check if x_dim of model fits data
        assert self.x_dim == data_train.shape[1], F'x_dim specified in model does not match data!' \
                                                  F'Should be {data_train.shape[1]}!'

        train_dataset = TensorDataset(torch.Tensor(data_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(data_val), torch.Tensor(y_val))

        dataloader_train = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=6)
        dataloader_val = DataLoader(val_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=6)

        return dataloader_train, dataloader_val

def main(argv):
    # init wandb logging
    config = dict(
        seed=FLAGS.seed,
        learning_rate=FLAGS.gen_lr,
        batch_size=FLAGS.gen_batch_size,
        hidden_size=FLAGS.dim_rnn_h,
        dataset="MIMIC-III",
        model=FLAGS.gen_model,
        pred_task="vent_bin"
    )

    use_cuda = torch.cuda.is_available()

    wandb.init(
        project='wand_project',
        entity='wandb_user',
        group='SRNN',
        job_type='cluster' if use_cuda else 'local',
        mode='online' if use_cuda else 'offline',
        config=config
    )

    generator = SRNNGenModel(seed=FLAGS.seed, x_dim=FLAGS.x_dim, z_dim=FLAGS.z_dim,
                             activation=FLAGS.activation, dropout=FLAGS.dropout,
                             dense_x_h=FLAGS.dense_x_h, dim_rnn_h=FLAGS.dim_rnn_h,
                             num_rnn_h=FLAGS.num_rnn_h, dense_hx_g=FLAGS.dense_hx_g,
                             dim_rnn_g=FLAGS.dim_rnn_g, num_rnn_g=FLAGS.num_rnn_g,
                             dense_gz_z=FLAGS.dense_gz_z, dense_hz_z=FLAGS.dense_hz_z,
                             dense_hz_x=FLAGS.dense_hz_x, beta=FLAGS.beta)
    generator.train_model()

if __name__ == '__main__':
    app.run(main)