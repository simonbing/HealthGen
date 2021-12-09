"""
2021 Simon Bing, ETHZ, MPI IS

Application to perform evaluation
"""
import healthgen.apps.global_parameters
from healthgen.apps.base_app import BaseApplication
from healthgen.apps.utils import split_labels_per_class
from healthgen.eval import GRUDEvaluator
import torch
import wandb
import logging
import os
import csv
from absl import flags, app

FLAGS = flags.FLAGS

class EvalApplication(BaseApplication):
    def __init__(self):
        super().__init__()
        self.eval_model = self._get_evaluator(eval_mode='synth')

    def _get_evaluator(self, eval_mode):
        if FLAGS.evaluation == 'grud':
            return GRUDEvaluator(seed=FLAGS.seed, eval_mode=eval_mode, batch_size=FLAGS.eval_batch_size,
                                 hidden_size=FLAGS.eval_hidden_size, num_layers=FLAGS.grud_num_layers,
                                 dropout=FLAGS.grud_dropout)

    def run(self):
        # Load real data
        X_real, y_real = self.data_loader.get_data()  # X: {dict}, y: {dict}

        X_train, X_val, _, y_train, y_val, _ = self.eval_model.get_data()
        self.eval_model.train_evaluator(X_train, X_val, y_train, y_val)

        _, _, X_test, _, _, y_test = self.eval_model.get_data(X_real, y_real)

        eval_score = self.eval_model.evaluate(X_test, y_test)

        # Aggregate scores from bagging
        if FLAGS.bagging:
            base_path, _ = os.path.split(FLAGS.out_path)
            bag_file_path = os.path.join(base_path, 'bootstrap_agg.csv')

            if os.path.exists(bag_file_path):
                write_mode = 'a'
            else:
                write_mode = 'w'

            with open(bag_file_path, write_mode) as f:
                writer = csv.writer(f)
                writer.writerow([eval_score])

        if FLAGS.multi_metric:
            wandb.run.summary.update({'eval_score': eval_score['auroc'],
                                      'sensitivity': eval_score['sens'],
                                      'specificity': eval_score['spec'],
                                      'bACC sens': eval_score['bacc_sens'],
                                      'f1_pos sens': eval_score['f1_sens'],
                                      'bACC opt': eval_score['bacc_opt'],
                                      'f1_pos opt': eval_score['f1_opt']})
        elif FLAGS.ROC_per_class is not None:
            if FLAGS.ROC_per_class == 'gender':
                feature_map = ['female', 'male']
            elif FLAGS.ROC_per_class == 'age':
                feature_map = ['<30', '31-50', '51-70', '>70']
            elif FLAGS.ROC_per_class == 'ethnicity':
                feature_map = ['white', 'black', 'hisp', 'asian', 'other']
            elif FLAGS.ROC_per_class == 'insurance':
                feature_map = ['medicare', 'medicaid', 'private', 'government', 'self']

            X_split, y_split = split_labels_per_class(X_test, y_test,
                                                      y_real['c_test'],
                                                      y_real['feature_names'],
                                                      stat_feat=FLAGS.ROC_per_class)
            for num, X_test_split in enumerate(X_split):
                if feature_map[num] == 'asian':
                    pass
                else:
                    eval_score_split = self.eval_model.evaluate(X_test_split, y_split[num])
                    wandb.run.summary.update({F'ROC {feature_map[num]}': eval_score_split})

                    if FLAGS.bagging:
                        base_path, _ = os.path.split(FLAGS.out_path)
                        bag_file_path = os.path.join(base_path, F'bootstrap_agg_{feature_map[num]}.csv')

                        if os.path.exists(bag_file_path):
                            write_mode = 'a'
                        else:
                            write_mode = 'w'

                        with open(bag_file_path, write_mode) as f:
                            writer = csv.writer(f)
                            writer.writerow([eval_score_split])

            wandb.run.summary.update({'eval_score': eval_score})
        else:
            wandb.run.summary.update({'eval_score': eval_score})
            logging.info(F'Evaluation score: {eval_score}.')

def main(argv):
    # init wandb logging
    config = dict(
        seed=FLAGS.seed,
        subgroup=FLAGS.subgroup,
        learning_rate=FLAGS.grud_lr,
        batch_size=FLAGS.eval_batch_size,
        hidden_size=FLAGS.eval_hidden_size,
        num_layers=FLAGS.grud_num_layers,
        dataset=FLAGS.dataset,
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

    application = EvalApplication()
    application.run()


if __name__ == '__main__':
    app.run(main)