"""
2021 Simon Bing, ETHZ, MPI IS
"""
import os
import numpy as np
import logging
from absl import flags, app
from healthgen.eval.base_evaluator import BaseEvaluator
from healthgen.eval.models.gru_d.gru_d_model import gru_d_model
import torch
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import wandb
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, confusion_matrix, f1_score

FLAGS=flags.FLAGS
flags.DEFINE_integer('eval_batch_size', 64, 'Batch size for eval model training.')
flags.DEFINE_integer('eval_hidden_size', 64, 'Hidden size for eval model training.')
flags.DEFINE_integer('eval_epochs', 100, 'Number of epochs for evaluation training.')
flags.DEFINE_integer('grud_num_layers', 1, 'Number of RNN layers in GRU-D')
flags.DEFINE_integer('grud_eval_step', 10, 'Steps after which to perform evaluation in GRU-D training.')
flags.DEFINE_float('grud_dropout', 0.1, 'Dropout in GRU-D training.')
flags.DEFINE_float('grud_lr', 0.0005, 'Learning rate decay for GRU-D training.')
flags.DEFINE_integer('grud_lr_decay_step', 20, 'Steps after which to apply learning rate decay for GRU-D.')
flags.DEFINE_float('grud_l2', 0.001, 'l2 penalty for GRU-D training.')
flags.DEFINE_string('eval_model_path', None, 'Previously trained and saved model for evaluation')
flags.DEFINE_bool('masks_only', False, 'Whether or not to only use the masks as input to the model.')
flags.DEFINE_bool('multi_metric', False, 'Whether or not to evaluate more metrics than AUROC.')
flags.DEFINE_enum('ROC_per_class', None, ['gender', 'age', 'ethnicity', 'insurance'], 'Class to compute split ROC scores on. '
                                                                                      'None for no per class eval.')



class GRUDEvaluator(BaseEvaluator):
    def __init__(self, seed, eval_mode, batch_size, hidden_size, num_layers, dropout):
        super().__init__(seed, eval_mode)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        # Set the seed for torch
        torch.manual_seed(self.seed)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def prepare_data(self, X_dict, y_dict):
        """
        TODO: fill in
        """
        if FLAGS.masks_only:
            print("Masks only eval!")

            X_train = np.stack((np.zeros_like(X_dict['X_train']), X_dict['m_train'], X_dict['delta_t_train']),
                                axis=1)
            X_val = np.stack((np.zeros_like(X_dict['X_val']), X_dict['m_val'], X_dict['delta_t_val']),
                              axis=1)
            X_test = np.stack((np.zeros_like(X_dict['X_test']), X_dict['m_test'], X_dict['delta_t_test']),
                               axis=1)
        else:
            X_train = np.stack((X_dict['X_train'], X_dict['m_train'], X_dict['delta_t_train']),
                                  axis=1)
            X_val = np.stack((X_dict['X_val'], X_dict['m_val'], X_dict['delta_t_val']),
                                axis=1)
            X_test = np.stack((X_dict['X_test'], X_dict['m_test'], X_dict['delta_t_test']),
                                 axis=1)

        # Need input size to build model
        self.input_size = X_train.shape[2]

        if len(y_dict['y_train'].shape) == 1:
            return X_train, X_val, X_test, y_dict['y_train'], y_dict['y_val'], y_dict['y_test']
        else:
            # Deals with synthetic labels that include static vars information
            return X_train, X_val, X_test, y_dict['y_train'][:,0], y_dict['y_val'][:,0], y_dict['y_test'][:,0]

    def _reshape_for_gru_d(self, X):
        """
        Args:
            X: dictionary containing features, missingness mask and delta_t arrays
        Retruns:
            train_data, val_data, test_data: [N_patients, [input, m_mask, delta_t], features, time_len]
        """
        train_data = np.stack((X['X_train'], X['m_train'], X['delta_t_train']),
                              axis=1)
        val_data = np.stack((X['X_val'], X['m_val'], X['delta_t_val']), axis=1)
        test_data = np.stack((X['X_test'], X['m_test'], X['delta_t_test']),
                             axis=1)

        return train_data, val_data, test_data

    def load_evaluator(self, saved_eval_model_path):
        # Build model
        self.model = gru_d_model(seed=self.seed, input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 output_size=1, num_layers=self.num_layers,
                                 dropout=self.dropout)
        # Load state dict
        self.model.load_state_dict(torch.load(saved_eval_model_path))
        self.model = self.model.to(self.device)


    def train_evaluator(self, X_train, X_val, y_train, y_val):
        """
        Trains a new GRU-D evaluator.
        """
        # Make training output directory
        train_base_path = os.path.join(FLAGS.out_path, 'eval', 'grud',
                                       self.eval_mode, 'training')
        if not os.path.exists(train_base_path):
            os.makedirs(train_base_path)

        # Prepare data for training
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val),torch.Tensor(y_val))
        x_mean = torch.Tensor(X_train[:,0,:,:].mean(axis=(0,2)))

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=True)

        # Build model
        # input_size = X_train.shape[2]
        self.model = gru_d_model(seed=self.seed, input_size=self.input_size, hidden_size=self.hidden_size,
                                 output_size=1, num_layers=self.num_layers,
                                 x_mean=x_mean, dropout=self.dropout)
        logging.info('GRU-D model built!')

        # Prepare training
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
            logging.info('Eval running on GPU')
        else:
            logging.info('Eval running on CPU')

        n_epochs = FLAGS.eval_epochs
        criterion = torch.nn.BCELoss()
        learning_rate = FLAGS.grud_lr
        learning_rate_decay_step = FLAGS.grud_lr_decay_step
        l2_penalty = FLAGS.grud_l2
        eval_step = FLAGS.grud_eval_step



        # Training loop
        logging.info('Training of GRU-D started!')

        train_num = len(train_dataloader) * FLAGS.eval_batch_size
        val_num = len(val_dataloader) * FLAGS.eval_batch_size

        train_step = 0
        val_step = 0
        best_val_roc = 0
        best_val_loss = np.inf
        for epoch in range(n_epochs):
            logging.info(F'Started epoch {epoch} of GRU-D training!')
            if learning_rate_decay_step != 0:
                if epoch % learning_rate_decay_step == 0: # update learning rate every decay step
                    learning_rate = learning_rate / 2
                    logging.info(F'Updated GRU-D learning rate to {learning_rate}.')

            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                         weight_decay=l2_penalty)

            losses, acc = [], []
            label, pred = [], []
            train_loss = 0

            self.model.train()
            for train_data, train_label in train_dataloader:
                train_data, train_label = train_data.to(self.device), train_label.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass, take only last RNN output as prediction
                y_pred = self.model(train_data)[:,-1,:]
                y_pred = torch.squeeze(y_pred)

                # Compute loss
                loss = criterion(y_pred, train_label)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                acc.append(
                    torch.eq(
                        (torch.sigmoid(y_pred).data > 0.5).float(),
                        train_label)
                )

                train_loss += loss.item()
                train_step = train_step + 1

            # Log (normalized) training loss
            wandb.log({F'{self.eval_mode}_train_loss': train_loss/train_num,
                       'epoch': epoch})

            if epoch % eval_step == 0:
                logging.info(F'Validating GRU-D at epoch {epoch}!')
                preds_val = []
                labels_val = []

                val_loss = 0

                self.model.eval()
                for val_data, val_label in val_dataloader:
                    val_data, val_label = val_data.to(self.device), val_label.to(self.device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass, take only last RNN output as prediction
                    y_pred = self.model(val_data)[:,-1,:]
                    y_pred = torch.squeeze(y_pred)

                    # Compute loss
                    loss = criterion(y_pred, val_label)

                    preds_val = np.append(preds_val, y_pred.detach().cpu().numpy())
                    labels_val = np.append(labels_val, val_label.detach().cpu().numpy())

                    val_loss += loss.item()

                    val_step = val_step + 1

                # Log (normalized) validation loss
                wandb.log({F'{self.eval_mode}_val_loss': val_loss/val_num,
                           'epoch': epoch})

                val_roc = roc_auc_score(labels_val, preds_val)
                val_prc = average_precision_score(labels_val, preds_val)
                if val_roc >= best_val_roc:
                    best_model_state_dict = deepcopy(self.model.state_dict())
                    best_val_roc = val_roc
                    best_val_loss = val_loss
                    logging.info(F'New best model saved! ROC: {val_roc}')

                wandb.log({F'{self.eval_mode}_val_roc': val_roc,
                           F'{self.eval_mode}_val_prc': val_prc,
                           'epoch': epoch})

        best_model_path = os.path.join(FLAGS.out_path, 'eval', 'grud',
                                       self.eval_mode, 'grud_best_model.pt')
        torch.save(best_model_state_dict, best_model_path)

        # Re-load the best model state
        self.model.load_state_dict(best_model_state_dict)

        logging.info('GRU-D training successfully finished!')

    def evaluate(self, X_test, y_test):
        # Make output directory
        eval_base_path = os.path.join(FLAGS.out_path, 'eval', 'grud', self.eval_mode)
        if not os.path.exists(eval_base_path):
            os.makedirs(eval_base_path)

        # Evaluation preparation
        test_dataset = TensorDataset(torch.Tensor(X_test),
                                     torch.Tensor(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     shuffle=True)

        # Compute eval on trained model with test data
        logging.info('Evaluating GRU-D on test data!')
        self.model.eval()
        preds_test = []
        labels_test = []
        for test_data, test_labels in test_dataloader:
            test_data, val_label = test_data.to(
                self.device), test_labels.to(self.device)

            # Forward pass, take only last RNN output as prediction
            y_pred = self.model(test_data)[:, -1, :]
            y_pred = torch.squeeze(y_pred)

            preds_test = np.append(preds_test,
                                   y_pred.detach().cpu().numpy())
            labels_test = np.append(labels_test,
                                    test_labels.detach().cpu().numpy())

        auroc = roc_auc_score(labels_test, preds_test)

        if FLAGS.multi_metric:
            fpr, tpr, thresholds = roc_curve(labels_test, preds_test)
            tnr = 1 - fpr
            # Find index of closest value to given specificity
            fixed_spec = 0.9
            thresh_idx = np.argmin((tnr[1:] - fixed_spec)**2) + 1
            sens_thresh = thresholds[thresh_idx]
            # Threshold prediction values
            preds_test_binary_sens = np.where(preds_test >= sens_thresh, 1, 0)
            tn, fp, fn, tp = confusion_matrix(labels_test, preds_test_binary_sens).ravel()

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            # ppv = tp / (tp + fp)
            # npv = tn / (tn + fn)
            bacc_sens = (sens + spec) / 2
            f1_sens = f1_score(labels_test, preds_test_binary_sens)

            ### Optimize threshold for other metrics
            J = tpr - fpr
            opt_idx = np.argmax(J[1:]) + 1
            opt_thresh = thresholds[opt_idx]
            # Threshold prediction values
            preds_test_binary_opt = np.where(preds_test >= opt_thresh, 1, 0)
            tn, fp, fn, tp = confusion_matrix(labels_test, preds_test_binary_opt).ravel()

            sens_opt = tp / (tp + fn)
            spec_opt = tn / (tn + fp)
            bacc_opt = (sens_opt + spec_opt) / 2
            f1_opt = f1_score(labels_test, preds_test_binary_opt)

            metrics = {'auroc': auroc, 'sens': sens, 'spec': spec, 'bacc_sens': bacc_sens,
                       'f1_sens': f1_sens, 'bacc_opt': bacc_opt, 'f1_opt': f1_opt}
            return metrics

        else:
            return auroc


def main(argv):
    # init wandb logging
    config = dict(
        seed = FLAGS.seed,
        learning_rate = FLAGS.grud_lr,
        batch_size = FLAGS.eval_batch_size,
        hidden_size = FLAGS.eval_hidden_size,
        num_layers = FLAGS.grud_num_layers,
        dataset = "physionet",
        model = "GRU-D",
        pred_task = "vent_bin"
    )

    use_cuda = torch.cuda.is_available()

    wandb.init(
        project = 'wand_project',
        entity = 'wandb_user',
        group = 'GRU-D Physionet',
        job_type = 'cluster' if use_cuda else 'local',
        mode='online' if use_cuda else 'offline',
        config = config
    )

    evaluator = GRUDEvaluator(seed=FLAGS.seed, batch_size=FLAGS.eval_batch_size, hidden_size=FLAGS.eval_hidden_size,
                              num_layers=FLAGS.grud_num_layers, dropout=FLAGS.grud_dropout)

    # Get data
    X_train, X_val, X_test, y_train, y_val, y_test = evaluator.get_data()
    # Train evaluator

    # Or load evaluator

    # Evaluate

    eval_score = evaluator.evaluate()
    logging.info(F'Evaluation finished! AUROC: {eval_score}.')

if __name__ == '__main__':
    app.run(main)



