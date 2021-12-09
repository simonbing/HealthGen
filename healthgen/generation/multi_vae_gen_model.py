"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from healthgen.generation.vae_gen_model import VAEGenModel
from healthgen.generation.models import MultiVAE
from absl import flags, app

FLAGS=flags.FLAGS

# Model
flags.DEFINE_enum('decoder', 'mlp', ['mlp', 'conv'], 'Which decoder architecture to use (for feats).')

class MultiVAEGenModel(VAEGenModel):
    def __init__(self, seed, x_dim, z_dim, seq_len, activation, dropout, encoder, decoder, dense_x_z,
                 dense_z_x, conv_x_z, conv_z_x, beta, data_mode, mask_loss):
        self.decoder = decoder
        super().__init__(seed, x_dim, z_dim, seq_len, activation, dropout, encoder, dense_x_z,
                 dense_z_x, conv_x_z, conv_z_x, beta, data_mode, mask_loss)

    def build_model(self):
        model = MultiVAE(x_dim=self.x_dim, z_dim=self.z_dim, seq_len=self.seq_len, activation=self.activation,
                    dropout_p=self.dropout_p, encoder=self.encoder, decoder=self.decoder, dense_x_z=self.dense_x_z,
                    dense_z_x=self.dense_z_x, conv_x_z=self.conv_x_z, conv_z_x=self.conv_z_x,
                    beta=self.beta, data_mode=self.data_mode, mask_loss=self.mask_loss, device=self.device).to(self.device)

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