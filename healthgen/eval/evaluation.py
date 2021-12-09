"""
2021 Simon Bing, ETHZ, MPI IS

Script to only perform evaluation
"""
import os
import numpy as np
import logging
import torch
import wandb
from healthgen.eval import GRUDEvaluator
from absl import flags, app

FLAGS = flags.FLAGS

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
        pred_task = "ve nt_bin"
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

    if FLAGS.eval_model == 'grud':
        evaluator = GRUDEvaluator(seed=FLAGS.seed, eval_mode=FLAGS.eval_mode,
                                  batch_size=FLAGS.eval_batch_size, hidden_size=FLAGS.eval_hidden_size,
                                  num_layers=FLAGS.grud_num_layers, dropout=FLAGS.grud_dropout)
    else:
        logging.error(F'The evaluation model {FLAGS.eval_model} is not defined!')

    # Get data
    X_train, X_val, X_test, y_train, y_val, y_test = evaluator.get_data()
    # Train evaluator
    if FLAGS.eval_model_path is None:
        evaluator.train_evaluator(X_train, X_val, y_train, y_val)
    # Or load evaluator
    else:
        evaluator.load_evaluator(saved_eval_model_path=FLAGS.eval_model_path)
    # Evaluate
    eval_score = evaluator.evaluate(X_test, y_test)
    logging.info(F'Evaluation finished! AUROC: {eval_score}.')

if __name__ == '__main__':
    app.run(main)