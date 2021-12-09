"""
2021 Simon Bing, ETHZ, MPI IS

Application to train generative model.
"""
import numpy as np
import healthgen.apps.global_parameters
from healthgen.apps.base_app import BaseApplication
from healthgen.generation import VAEGenModel, MultiVAEGenModel, SRNNGenModel, KVAEGenModel, KVAEMissGenModel, HealthGenModel
import torch
import wandb
from absl import flags, app

FLAGS = flags.FLAGS

class GenApplication(BaseApplication):
    def __init__(self):
        super().__init__()
        self.gen_model = self._get_gen_model()

    def _get_gen_model(self):
        if FLAGS.gen_model == 'vae':
            return VAEGenModel(seed=FLAGS.seed, x_dim=FLAGS.x_dim, z_dim=FLAGS.z_dim, seq_len=FLAGS.seq_len,
                               activation=FLAGS.activation, dropout=FLAGS.dropout,
                               dense_x_z=FLAGS.dense_x_z, dense_z_x=FLAGS.dense_z_x,
                               conv_x_z=FLAGS.conv_x_z, conv_z_x=FLAGS.conv_z_x,
                               encoder=FLAGS.encoder, beta=FLAGS.beta, data_mode=FLAGS.data_mode,
                               mask_loss=FLAGS.mask_loss)
        elif FLAGS.gen_model == 'multi_vae':
            return MultiVAEGenModel(seed=FLAGS.seed, x_dim=FLAGS.x_dim, z_dim=FLAGS.z_dim, seq_len=FLAGS.seq_len,
                                    activation=FLAGS.activation, dropout=FLAGS.dropout,
                                    dense_x_z=FLAGS.dense_x_z, dense_z_x=FLAGS.dense_z_x,
                                    conv_x_z=FLAGS.conv_x_z, conv_z_x=FLAGS.conv_z_x,
                                    encoder=FLAGS.encoder, decoder=FLAGS.decoder, beta=FLAGS.beta, data_mode=FLAGS.data_mode,
                                    mask_loss=FLAGS.mask_loss)
        elif FLAGS.gen_model == 'healthgen':
            return HealthGenModel(seed=FLAGS.seed, x_dim=FLAGS.x_dim, y_dim=FLAGS.y_dim, v_dim=FLAGS.v_dim,
                                  z_dim=FLAGS.z_dim, seq_len=FLAGS.seq_len, activation=FLAGS.activation,
                                  dropout=FLAGS.dropout, dense_x_v=FLAGS.dense_x_v, dense_x_h=FLAGS.dense_x_h,
                                  dense_hx_g=FLAGS.dense_hx_g, dense_gz_z=FLAGS.dense_gz_z,
                                  dim_RNN_h=FLAGS.dim_rnn_h, num_RNN_h=FLAGS.num_rnn_h, dim_RNN_g=FLAGS.dim_rnn_g,
                                  num_RNN_g=FLAGS.num_rnn_g, dense_hz_z=FLAGS.dense_hz_z, dense_hz_x=FLAGS.dense_hz_x, dense_v_m=FLAGS.dense_v_m,
                                  beta=FLAGS.beta)
        elif FLAGS.gen_model == 'srnn':
            return SRNNGenModel(seed=FLAGS.seed, x_dim=FLAGS.x_dim, z_dim=FLAGS.z_dim,
                                activation=FLAGS.activation, dropout=FLAGS.dropout,
                                dense_x_h=FLAGS.dense_x_h, dim_rnn_h=FLAGS.dim_rnn_h,
                                num_rnn_h=FLAGS.num_rnn_h, dense_hx_g=FLAGS.dense_hx_g,
                                dim_rnn_g=FLAGS.dim_rnn_g, num_rnn_g=FLAGS.num_rnn_g,
                                dense_gz_z=FLAGS.dense_gz_z, dense_hz_z=FLAGS.dense_hz_z,
                                dense_hz_x=FLAGS.dense_hz_x, beta=FLAGS.beta)
        elif FLAGS.gen_model == 'kvae':
            return KVAEGenModel(seed=FLAGS.seed, u_dim=FLAGS.y_dim, x_dim=FLAGS.x_dim, a_dim=FLAGS.a_dim,
                                z_dim=FLAGS.z_dim, activation=FLAGS.activation,
                                dropout=FLAGS.dropout, dense_x_a=FLAGS.dense_x_a,
                                dense_a_x=FLAGS.dense_a_x, init_kf_mat=FLAGS.init_kf_mat,
                                noise_transition=FLAGS.noise_transition,
                                noise_emission=FLAGS.noise_emission, init_cov=FLAGS.init_cov,
                                K=FLAGS.K, dim_rnn_alpha=FLAGS.dim_rnn_alpha,
                                num_rnn_alpha=FLAGS.num_rnn_alpha, scale_recon=FLAGS.scale_recon,
                                use_smoothed_a=FLAGS.use_smoothed_a)
        elif FLAGS.gen_model == 'kvae_miss':
            return KVAEMissGenModel(seed=FLAGS.seed, u_dim=FLAGS.u_dim, x_dim=FLAGS.x_dim, m_dim=FLAGS.m_dim, a_dim=FLAGS.a_dim,
                                z_dim=FLAGS.z_dim, activation=FLAGS.activation,
                                dropout=FLAGS.dropout, dense_x_a=FLAGS.dense_x_a,
                                dense_a_x=FLAGS.dense_a_x, init_kf_mat=FLAGS.init_kf_mat,
                                noise_transition=FLAGS.noise_transition,
                                noise_emission=FLAGS.noise_emission, init_cov=FLAGS.init_cov,
                                K=FLAGS.K, dim_rnn_alpha=FLAGS.dim_rnn_alpha,
                                num_rnn_alpha=FLAGS.num_rnn_alpha, scale_recon=FLAGS.scale_recon,
                                use_smoothed_a=FLAGS.use_smoothed_a, sample_m=FLAGS.sample_m,
                                learn_scale=FLAGS.learn_scale)

    def _get_cond_labels(self, y_dict):
        """
        Returns: cond_labels, labels used for conditional generation.
        """
        cond_labels = np.concatenate((y_dict['y_train'], y_dict['y_val']))

    def run(self):
        # Load data
        X_real, y_real = self.data_loader.get_data()  # X: {dict}, y: {dict}

        # Train generative model
        self.gen_model.train_model(X_real, y_real)

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
        group=FLAGS.gen_model,
        job_type='cluster' if use_cuda else 'local',
        mode='online' if use_cuda else 'offline',
        config=config
    )

    application = GenApplication()
    application.run()


if __name__ == '__main__':
    app.run(main)

