"""
2021 Simon Bing, ETHZ, MPI IS

Main application to conduct experiments. Contains entire pipeline.
"""
import os
import sys
import numpy as np
from healthgen.apps.generation import GenApplication
from healthgen.eval import GRUDEvaluator
from healthgen.apps.utils import get_synth_labels, augment_data, split_labels_per_class
import logging
import torch
import wandb
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_bool('save_synth', False, 'Whether or not to save the synthetically generated data.')
flags.DEFINE_bool('eval_real', False, 'Whether or not to perform evlauation on the real data as well.')

class MainApplication(GenApplication):
    def __init__(self):
        super().__init__()
        if not os.path.isdir(FLAGS.out_path):
            os.makedirs(FLAGS.out_path)

        self.evaluator_real = self._get_evaluator(eval_mode='real')
        self.evaluator_synth = self._get_evaluator(eval_mode='synth')

    def _get_evaluator(self, eval_mode):
        if FLAGS.evaluation == 'grud':
            return GRUDEvaluator(seed=FLAGS.seed, eval_mode=eval_mode, batch_size=FLAGS.eval_batch_size,
                                 hidden_size=FLAGS.eval_hidden_size, num_layers=FLAGS.grud_num_layers,
                                 dropout=FLAGS.grud_dropout)


    def run(self):
        # Load data
        X_real, y_real = self.data_loader.get_data() # X: {dict}, y: {dict}

        ### Get generative model ###
        if FLAGS.saved_gen_model is not None:
            # Load model
            self.gen_model.model.load_state_dict(torch.load(FLAGS.saved_gen_model,
                                                  map_location=self.gen_model.device))
            logging.info(F'Loaded previously trained {FLAGS.gen_model} model!')
        else:
            # Train model
            self.gen_model.train_model(X_real, y_real)

        if FLAGS.debug:
            # Save reconstructions (of validation data)
            data_orig, data_recon = self.gen_model.reconstruct(X_real, y_real, N=200)
            np.save(os.path.join(FLAGS.out_path, 'gen', FLAGS.gen_model, 'data_orig.npy'), data_orig)
            np.save(os.path.join(FLAGS.out_path, 'gen', FLAGS.gen_model, 'data_recon.npy'), data_recon)

        ### Generate synthetic data ###
        y_all = get_synth_labels(y_real, mode=FLAGS.gen_mode, split=FLAGS.split,
                                 cond_static=FLAGS.cond_static, static_vars=FLAGS.static_vars)
        X_synth, y_synth = self.gen_model.generate_cond(labels=y_all, seq_len=25)
        # Augment real data with synthetic if needed
        if FLAGS.gen_mode == 'augment':
            X_synth, y_synth = augment_data(X_real, y_real, X_synth, y_synth)

        if FLAGS.save_synth:
            np.savez(os.path.join(FLAGS.out_path, 'gen', FLAGS.gen_model, 'X_synth.npz'), **X_synth)
            np.savez(os.path.join(FLAGS.out_path, 'gen', FLAGS.gen_model, 'y_synth.npz'), **y_synth)

        ### Evaluation ###
        X_synth_train, X_synth_val, X_synth_test, \
        y_synth_train, y_synth_val, y_synth_test = self.evaluator_synth.get_data(X_synth, y_synth)

        X_real_train, X_real_val, X_real_test, \
        y_real_train, y_real_val, y_real_test = self.evaluator_real.get_data(X_real, y_real)

        # Train evaluator on synthetic data
        self.evaluator_synth.train_evaluator(X_synth_train, X_synth_val, y_synth_train, y_synth_val)
        eval_score_synth = self.evaluator_synth.evaluate(X_real_test, y_real_test)

        if FLAGS.multi_metric:
            wandb.run.summary.update({'eval_score': eval_score_synth['auroc'],
                                      'sensitivity': eval_score_synth['sens'],
                                      'specificity': eval_score_synth['spec'],
                                      'bACC sens': eval_score_synth['bacc_sens'],
                                      'f1_pos sens': eval_score_synth['f1_sens'],
                                      'bACC opt': eval_score_synth['bacc_opt'],
                                      'f1_pos opt': eval_score_synth['f1_opt']})
        elif FLAGS.ROC_per_class is not None:
            if FLAGS.ROC_per_class == 'gender':
                feature_map = ['female', 'male']
            elif FLAGS.ROC_per_class == 'age':
                feature_map = ['<30', '31-50', '51-70', '>70']
            elif FLAGS.ROC_per_class == 'ethnicity':
                feature_map = ['white', 'black', 'hisp', 'asian', 'other']
            elif FLAGS.ROC_per_class == 'insurance':
                feature_map = ['medicare', 'medicaid', 'private', 'government', 'self']

            X_split, y_split = split_labels_per_class(X_real_test, y_real_test,
                                                      y_real['c_test'],
                                                      y_real['feature_names'],
                                                      stat_feat=FLAGS.ROC_per_class)
            for num, X_test_split in enumerate(X_split):
                if feature_map[num] == 'asian':
                    pass
                else:
                    eval_score_split = self.evaluator_synth.evaluate(X_test_split, y_split[num])
                    wandb.run.summary.update({F'ROC {feature_map[num]}': eval_score_split})

            wandb.run.summary.update({'eval_score': eval_score_synth})
        else:
            logging.info(F'Evaluation score on synthetic data: {eval_score_synth}.')
            wandb.run.summary.update({'eval_score_synth': eval_score_synth})



        if FLAGS.eval_real:
            if FLAGS.eval_model_path is not None:
                self.evaluator_real.load_evaluator(FLAGS.eval_model_path)
            else:
                # Train evaluator on real data
                self.evaluator_real.train_evaluator(X_real_train, X_real_val, y_real_train, y_real_val)
            eval_score_real = self.evaluator_real.evaluate(X_real_test, y_real_test)
            logging.info(F'Evaluation score on real data: {eval_score_real}.')
            wandb.run.summary.update({'eval_score_real': eval_score_real})

        logging.info('Experiment successfully finished!')


def main(argv):
    # init wandb logging
    config = dict(
        seed=FLAGS.seed,
        subgroup=FLAGS.subgroup,
        learning_rate=FLAGS.gen_lr,
        batch_size=FLAGS.gen_batch_size,
        hidden_size=FLAGS.eval_hidden_size,
        num_layers=FLAGS.grud_num_layers,
        dataset=FLAGS.dataset,
        eval_model=FLAGS.evaluation,
        gen_model=FLAGS.gen_model,
        pred_task="vent_bin"
    )

    use_cuda = torch.cuda.is_available()
    # Check if we are in debug mode
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace():
        mode = 'offline'
    else:
        mode = 'online'

    wandb.init(
        project='wand_project',
        entity='wandb_user',
        group=FLAGS.group,
        job_type='cluster' if use_cuda else 'local',
        mode=mode,
        config=config
    )

    if FLAGS.run_name is not None:
        wandb.run.name = FLAGS.run_name

    application = MainApplication()
    application.run()


if __name__ == '__main__':
    app.run(main)