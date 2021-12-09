"""
2021 Simon Bing, ETHZ, MPI IS

Application to perform evaluation
"""
import healthgen.apps.global_parameters
from healthgen.apps.base_app import BaseApplication
from healthgen.eval import GRUDEvaluator
import torch
import wandb
import logging
import os
import csv
import numpy as np
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('X_id_path', '', 'Path to in distribution training data.')
flags.DEFINE_string('y_id_path', '', 'Path to in distribution training labels.')
flags.DEFINE_string('X_ood_path', '', 'Path to out of distribution training data.')
flags.DEFINE_string('y_ood_path', '', 'Path to out of distribution training labels.')

class RobustEvalApplication(BaseApplication):
    def __init__(self):
        super().__init__()
        self.eval_model = self._get_evaluator(eval_mode='synth')

    def _get_evaluator(self, eval_mode):
        if FLAGS.evaluation == 'grud':
            return GRUDEvaluator(seed=FLAGS.seed, eval_mode=eval_mode, batch_size=FLAGS.eval_batch_size,
                                 hidden_size=FLAGS.eval_hidden_size, num_layers=FLAGS.grud_num_layers,
                                 dropout=FLAGS.grud_dropout)


    def run(self):
        # Load (in distribution) training data
        X_id = np.load(FLAGS.X_id_path)
        y_id = np.load(FLAGS.y_id_path)
        X_id_train, X_id_val, X_id_test, y_id_train, y_id_val, y_id_test = self.eval_model.get_data(X_id, y_id)

        # Train evaluation model OR load pretrained model
        if FLAGS.eval_model_path is None:
            self.eval_model.train_evaluator(X_id_train, X_id_val, y_id_train, y_id_val)
        else:
            self.eval_model.load_evaluator(FLAGS.eval_model_path)

        # Evaluate model on training (in distribution) test set
        eval_score_id = self.eval_model.evaluate(X_id_test, y_id_test)

        # Load (out of distribution) test data
        X_ood = np.load(FLAGS.X_ood_path)
        y_ood = np.load(FLAGS.y_ood_path)
        _, _, X_ood_test, _, _, y_ood_test = self.eval_model.get_data(X_ood, y_ood)

        # Evaluate model on test set
        eval_score_ood = self.eval_model.evaluate(X_ood_test, y_ood_test)

        # Log scores
        if FLAGS.multi_metric:
            wandb.run.summary.update({'eval_score id': eval_score_id['auroc'],
                                      'sensitivity id': eval_score_id['sens'],
                                      'specificity id': eval_score_id['spec'],
                                      'bACC sens id': eval_score_id['bacc_sens'],
                                      'f1 sens id': eval_score_id['f1_sens'],
                                      'bACC opt id': eval_score_id['bacc_opt'],
                                      'f1 opt id': eval_score_id['f1_opt']})
            wandb.run.summary.update({'eval_score ood': eval_score_ood['auroc'],
                                      'sensitivity ood': eval_score_ood['sens'],
                                      'specificity ood': eval_score_ood['spec'],
                                      'bACC sens ood': eval_score_ood['bacc_sens'],
                                      'f1 sens ood': eval_score_ood['f1_sens'],
                                      'bACC opt ood': eval_score_ood['bacc_opt'],
                                      'f1 opt ood': eval_score_ood['f1_opt']})
        else:
            wandb.run.summary.update({'eval_score_id': eval_score_id})
            logging.info(F'Evaluation score i.d.: {eval_score_id}.')
            wandb.run.summary.update({'eval_score_ood': eval_score_ood})
            logging.info(F'Evaluation score o.o.d.: {eval_score_ood}.')


def main(argv):
    # init wandb logging
    config = dict(
        seed=FLAGS.seed,
        subgroup=FLAGS.subgroup,
        eval_model=FLAGS.evaluation,
        pred_task="vent_bin"
    )

    use_cuda = torch.cuda.is_available()

    wandb.init(
        project='wand_project',
        entity='wandb_user',
        group=FLAGS.group,
        job_type='cluster' if use_cuda else 'local',
        mode='online' if use_cuda else 'offline',
        config=config
    )

    if FLAGS.run_name is not None:
        wandb.run.name = FLAGS.run_name

    application = RobustEvalApplication()
    application.run()


if __name__ == '__main__':
    app.run(main)