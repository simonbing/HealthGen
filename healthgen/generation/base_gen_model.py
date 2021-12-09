"""
2021 Simon Bing, ETHZ, MPI IS
"""
import six
import numpy as np
import os
import healthgen.apps.global_parameters
from healthgen.apps.utils import plot_reconstructions
from healthgen.data_access.preprocessing import MissingnessDeltaT
import logging
import torch
from torch.distributions.bernoulli import Bernoulli
from copy import deepcopy
from sklearn.model_selection import train_test_split
import wandb
from abc import ABCMeta, abstractmethod
from absl import flags

FLAGS = flags.FLAGS
# Training Data
flags.DEFINE_string('X_train_path', '', 'Path to training data for generative model.')
flags.DEFINE_string('y_train_path', '', 'Path to training labels for generative model.')
flags.DEFINE_enum('data', 'mimic', ['mimic', 'ball_box'], 'Dataset for training')
flags.DEFINE_enum('data_mode', 'all', ['all', 'feats', 'mask', 'feats_mask'],
                  'Which mode of the training data to use i.e. with missingness mask'
                  'and delta_t, or only the input features.')
flags.DEFINE_bool('cond_static', False, 'Whether or not to condition on static variables.')
flags.DEFINE_list('static_vars', None, 'List of static variables to condition on.')
# Model
flags.DEFINE_enum('gen_model', None, ['srnn', 'kvae', 'kvae_miss', 'vae', 'multi_vae', 'healthgen'], 'Generative model to use.')
flags.DEFINE_bool('print_model', False, 'Whether or not to print a model summary.')
flags.DEFINE_string('saved_gen_model', None, 'Path to previously trained and saved generative model.')
flags.DEFINE_float('beta', 1.0, 'Tradeoff parameter for beta-VAE')
# Shared model parameters
flags.DEFINE_integer('x_dim', 274, 'Input dimension.')
flags.DEFINE_integer('y_dim', 1, 'External input dimension.')
flags.DEFINE_integer('z_dim', 16, 'Latent space dimension.')
flags.DEFINE_integer('seq_len', 25, 'Length of the input sequences.')
flags.DEFINE_enum('activation', 'tanh', ['tanh', 'relu', 'leaky_relu'], 'Activation function.')
flags.DEFINE_float('dropout', 0.0, 'Dropout percentage.')
# Training
flags.DEFINE_integer('gen_epochs', 100, 'Number of training epochs for generative model.')
flags.DEFINE_integer('gen_batch_size', 64, 'Batch size for generative model training.')
flags.DEFINE_float('gen_lr', 0.001, 'Learning rate for generative model training.')
flags.DEFINE_integer('gen_lr_decay_step', 0, 'Steps after which to apply learning rate decay for gen model.')
flags.DEFINE_integer('gen_val_freq', 10, 'Validation freuquency for generative model training')
flags.DEFINE_integer('gen_early_stop_patience', 20, 'Number of epochs of patience for early stopping.')
flags.DEFINE_bool('mask_loss', False, 'Whether or not to mask the loss with the missingness masks.')
# Generation
flags.DEFINE_enum('gen_mode', 'synth', ['synth', 'augment'], 'Whether to generate fully synthetic data or augment real data.')
flags.DEFINE_float('split', None, 'Split for labels for conditional generation.')
flags.DEFINE_bool('resample_m_delta_t', False, 'Whether or not to resample the missingness'
                                               'and delta_t masks from the generated data.')
flags.DEFINE_bool('deterministic_gen', False, 'Whether or not to add stochastic noise to data generation.')
flags.DEFINE_bool('mask_gen_data', False, 'Whether or not to mask generated features with generated masks')

flags.mark_flag_as_required('gen_model')

@six.add_metaclass(ABCMeta)
class BaseGenModel(object):
    def __init__(self, seed):
        self.seed = seed
        self.randomstate = np.random.RandomState(self.seed)

        # torch params
        torch.manual_seed(self.seed)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        # Build model
        self.model = self.build_model()

        # Directory for outputs
        self.gen_out_path = os.path.join(FLAGS.out_path, 'gen', FLAGS.gen_model)
        if not os.path.isdir(self.gen_out_path):
            os.makedirs(self.gen_out_path)

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def build_dataloader(self, X, y, c, batch_size):
        raise NotImplementedError

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=FLAGS.gen_lr)

    def train_loop(self, train_dataloader, val_dataloader):
        """
        Training loop of generative model.
        """
        logging.info(F'Training of {FLAGS.gen_model} started!')
        # Training preparation
        train_out_path = os.path.join(self.gen_out_path, 'training')
        if not os.path.isdir(train_out_path):
            os.makedirs(train_out_path)

        train_num = len(train_dataloader) * FLAGS.gen_batch_size
        val_num = len(val_dataloader) * FLAGS.gen_batch_size

        best_val_loss = np.inf
        best_epoch = 0
        stop_patience = 0

        learning_rate_decay_step = FLAGS.gen_lr_decay_step

        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        # Train with mini-batch SGD
        for epoch in range(FLAGS.gen_epochs):
            # Batch training
            train_loss_tot = 0
            train_loss_recon = 0
            train_loss_KL = 0
            if learning_rate_decay_step != 0:
                # Learning rate scheduler
                milestones = np.arange(learning_rate_decay_step, FLAGS.gen_epochs, learning_rate_decay_step)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.85)
            for train_batch, train_label in train_dataloader:
                train_batch = train_batch.to(self.device)
                train_label = train_label.to(self.device)
                # Zero gradients
                self.optimizer.zero_grad()
                # Forward pass
                recon_train_batch = self.model(train_batch, train_label, compute_loss=True)
                # Compute loss
                loss_tot, loss_recon, loss_KL = self.model.loss
                # Backward pass
                loss_tot.backward()
                # Update weights
                self.optimizer.step()

                train_loss_tot += loss_tot.item()
                train_loss_recon += loss_recon.item()
                train_loss_KL += loss_KL.item()

            # Log (normalized) training losses
            wandb.log({'train_loss_tot': train_loss_tot / train_num,
                       'train_loss_recon': train_loss_recon / train_num,
                       'train_loss_KL': train_loss_KL / train_num,
                       'epoch': epoch})

            if epoch % FLAGS.gen_val_freq == 0:
                val_loss_tot = 0
                val_loss_recon = 0
                val_loss_KL = 0
                # self.model.eval() # for some reason this breaks on GPU
                for val_batch, val_label in val_dataloader:
                    val_batch = val_batch.to(self.device)
                    val_label = val_label.to(self.device)
                    # Forward pass
                    recon_val_batch = self.model(val_batch, val_label, compute_loss=True)
                    # Compute loss
                    loss_tot, loss_recon, loss_KL = self.model.loss

                    val_loss_tot += loss_tot.item()
                    val_loss_recon += loss_recon.item()
                    val_loss_KL += loss_KL.item()

                # Save reconstructions of training batch
                val_batch_np = val_batch.cpu().detach().numpy()
                # Sample from Bernoulli distribution if we only generate masks
                if FLAGS.data_mode == 'mask':
                    # recon_batch_distr = Bernoulli(probs=recon_val_batch)
                    # recon_val_batch = recon_batch_distr.sample()
                    # Or: threshold probabilities without sampling
                    # recon_val_batch = torch.round(recon_val_batch)
                    pass
                recon_val_batch_np = recon_val_batch.cpu().detach().numpy()
                # Save reconstruction arrays
                # in_batch_path = os.path.join(train_out_path, F'epoch_{epoch}_recon_in.npy')
                # recon_batch_path = os.path.join(train_out_path, F'epoch_{epoch}_recon_out.npy')
                # np.save(in_batch_path, val_batch_np)
                # np.save(recon_batch_path, recon_val_batch_np)
                # Log visualizations
                # recon_plot = plot_reconstructions(val_batch_np, recon_val_batch_np,
                #                      patient_idx=0, data_mode=FLAGS.data_mode)
                # wandb.log({'reconstructions': recon_plot, 'epoch': epoch})

                # Log (normalized) validation losses
                wandb.log({'val_loss_tot': val_loss_tot / val_num,
                           'val_loss_recon': val_loss_recon / val_num,
                           'val_loss_KL': val_loss_KL / val_num,
                           'epoch': epoch})
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
                    logging.info(F'Early stopping patience achieved. Stopping training '
                                 F'at epoch {epoch}!')
                    break

            if learning_rate_decay_step != 0:
                scheduler.step(epoch=epoch)

            # TODO: checkpointing during training
        # Save the best model
        best_model_path = os.path.join(train_out_path, F'{FLAGS.gen_model}_best_model.pt')
        logging.info(F'Saved best model from epoch {best_epoch}!')
        torch.save(best_model_state_dict, best_model_path)
        logging.info(F'Training of {FLAGS.gen_model} successfully finished!')

    def train_model(self, X=None, y=None):
        """
        Method to train generative models.

        Args:
            X, y: dicts containing training data and labels
        """
        # Get train data
        if X is None and y is None:
            X = np.load(FLAGS.X_train_path)
            y = np.load(FLAGS.y_train_path)

        # Build dataloaders
        train_dataloader, val_dataloader = self.build_dataloader(X, y, FLAGS.gen_batch_size)

        # Build model
        # self.model = self.build_model()
        print(F'Nr. of mini-batch steps for one epoch: {len(train_dataloader)}.')
        # EXPERIMENTAL, I think the model wasn't being moved to GPU...
        self.model = self.model.to(self.device)
        print(F'{FLAGS.gen_model} built on {self.device} device!')

        # Optionally print model summary
        if FLAGS.print_model:
            for log in self.model.get_info():
                logging.info(log)

        # Initialize optimizer
        self.init_optimizer()

        # Training loop
        self.train_loop(train_dataloader, val_dataloader)

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
        _, val_dataloader = self.build_dataloader(X, y, N)
        # Get single batch of size N from loader
        data_orig, labels_orig = next(iter(val_dataloader))
        data_orig, labels_orig = data_orig.to(self.device), labels_orig.to(self.device)

        with torch.no_grad():
            data_recon = self.model(data_orig, labels_orig, compute_loss=False)
            if FLAGS.data_mode == 'mask':
                m_distr = Bernoulli(probs=data_recon)
                data_recon = m_distr.sample().int().cpu().detach().numpy()
            else:
                data_recon = data_recon.cpu().detach().numpy()

        return data_orig.cpu().detach().numpy(), data_recon

    def generate_cond(self, labels, seq_len):
        """
        Method to conditionally generate data, given labels.

        Args:
            labels: list of labels for conditional generation [N,]. N determines
                    the number of time series to generate.
            seq_len: length of time series to generate (int)

        Returns:
            X_synth_dict: dict containing synthetically generated data, split
                          into train/val/test
            y_synth_dict: dict containing labels, split into train/val/test
        """
        print(F'Device in base_gen_model.py: {self.device}')
        self.model = self.model.to(self.device)
        X_synth, y_synth = self.model.generate_synth(labels=labels, seq_len=seq_len,
                                                     det_gen=FLAGS.deterministic_gen)

        if FLAGS.data == 'mimic':
            # Split data depending on data_mode
            if FLAGS.data_mode == 'all':
                X, m, delta_t = np.split(X_synth, 3, axis=1)
                if FLAGS.resample_m_delta_t:
                    delta_t = MissingnessDeltaT()._get_delta_t(m)
                    # X, m, delta_t = MissingnessDeltaT().transform(X).values()
                X_stack = np.stack((X, m, delta_t), axis=1)

            elif FLAGS.data_mode == 'feats_mask':
                X, m = np.split(X_synth, 2, axis=1)
                if FLAGS.mask_gen_data:
                    X = np.where(m, X, 0.0)
                delta_t = MissingnessDeltaT()._get_delta_t(m)
                X_stack = np.stack((X, m, delta_t), axis=1)
            elif FLAGS.data_mode == 'feats':
                X, m, delta_t = MissingnessDeltaT().transform(X_synth).values()
                X_stack = np.stack((X, m, delta_t), axis=1)
            elif FLAGS.data_mode == 'mask':
                m = X_synth
                X = np.zeros_like(m)
                delta_t = MissingnessDeltaT()._get_delta_t(m)
                X_stack = np.stack((X, m, delta_t), axis=1)
        elif FLAGS.data == 'ball_box':
            X = X_synth
            m = np.zeros_like(X)
            delta_t = np.zeros_like(X)
            X_stack = np.stack((X, m, delta_t), axis=1)

        # TODO: MOVE THIS TO UTIL FUNCTION
        val_fraction = 0.15
        test_fraction = 0.15
        train_fraction = 1.0 - val_fraction - test_fraction

        if len(y_synth.shape) == 1:  # stratify only if we have a single task
            y_strat = y_synth
        else:
            y_strat = None

        X_synth_train, X_intermed, y_synth_train, y_intermed = train_test_split(X_stack, y_synth,
                                                                test_size=1 - train_fraction,
                                                                random_state=self.randomstate,
                                                                stratify=y_strat)
        if y_strat is not None:
            y_intermed_strat = y_intermed
        else:
            y_intermed_strat = None
        X_synth_val, X_synth_test, y_synth_val, y_synth_test = train_test_split(X_intermed, y_intermed,
                                                        test_size=test_fraction / (test_fraction + val_fraction),
                                                        random_state=self.randomstate,
                                                        stratify=y_intermed_strat)
        X_synth_dict = {
            'X_train': X_synth_train[:,0,...],
            'X_val': X_synth_val[:,0,...],
            'X_test': X_synth_test[:,0,...],
            'm_train': X_synth_train[:,1,...],
            'm_val': X_synth_val[:,1,...],
            'm_test': X_synth_test[:,1,...],
            'delta_t_train': X_synth_train[:,2,...],
            'delta_t_val': X_synth_val[:,2,...],
            'delta_t_test': X_synth_test[:,2,...]
        }
        y_synth_dict = {
            'y_train' : y_synth_train,
            'y_val': y_synth_val,
            'y_test': y_synth_test,
        }

        return X_synth_dict, y_synth_dict