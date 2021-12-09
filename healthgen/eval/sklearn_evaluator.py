"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
import logging
from absl import flags, app
from healthgen.eval.base_evaluator import BaseEvaluator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

FLAGS = flags.FLAGS
flags.DEFINE_enum('sklearn_input_mode', 'all', ['all', 'feats_mask', 'feats'],
                  'Which inputs to include for training.')
flags.DEFINE_enum('sklearn_imputation', 'none', ['none', 'mean', 'forward'],
                  'Imputation scheme for evaluation data.')
flags.DEFINE_enum('sklearn_model', 'lr', ['lr', 'svm', 'rf'], 'Model for downstream evaluation')

class SKLearnEvaluator(BaseEvaluator):
    def __init__(self, seed, input_mode, imputation, classifier):
        super().__init__(seed)
        self.input_mode = input_mode
        self.imputation = imputation
        self.classifier = classifier

    def _impute(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """

        if self.imputation == 'mean':
            X_train_mean_missing = np.where(X['m_train'], X['X_train'], np.nan)
            a=0

        elif self.imputation == 'forward':
            pass
        else:
            raise ValueError("'imputation' must be on of ['none', 'mean', 'forward']!")

        return X

    def _reshape(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        if self.input_mode == 'all':
            X_train = np.concatenate((X['X_train'], X['m_train'], X['delta_t_train']), axis=1)
            X_val = np.concatenate((X['X_val'], X['m_val'], X['delta_t_val']), axis=1)
            X_test = np.concatenate((X['X_test'], X['m_test'], X['delta_t_test']), axis=1)
            X_train_reshape = np.reshape(X_train, (X_train.shape[0], -1))
            X_val_reshape = np.reshape(X_val, (X_val.shape[0], -1))
            X_test_reshape = np.reshape(X_test, (X_test.shape[0], -1))
        elif self.input_mode == 'feats_mask':
            X_train = np.concatenate((X['X_train'], X['m_train']), axis=1)
            X_val = np.concatenate((X['X_val'], X['m_val']), axis=1)
            X_test = np.concatenate((X['X_test'], X['m_test']), axis=1)
            X_train_reshape = np.reshape(X_train, (X_train.shape[0], -1))
            X_val_reshape = np.reshape(X_val, (X_val.shape[0], -1))
            X_test_reshape = np.reshape(X_test, (X_test.shape[0], -1))
        elif self.input_mode == 'feats':
            X_train_reshape = np.reshape(X['X_train'], (X['X_train'].shape[0], -1))
            X_val_reshape = np.reshape(X['X_val'], (X['X_val'].shape[0], -1))
            X_test_reshape = np.reshape(X['X_test'], (X['X_test'].shape[0], -1))
        else:
            raise ValueError("'input_mode' must be on of ['all', 'feats_mask', 'feats']!")

        return X_train_reshape, X_val_reshape, X_test_reshape

    def _get_classifier(self):
        if self.classifier == 'lr':
            clf = LogisticRegression(random_state=self.seed, max_iter=1000)
        elif self.classifier == 'svm':
            clf = SVC(random_state=self.seed)
        elif self.classifier == 'rf':
            clf = RandomForestClassifier(random_state=self.seed)

        return clf


    def do_eval(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        if self.imputation != 'none':
            X = self._impute(X)

        # Reshape data depending on selected mode
        X_train, X_val, X_test = self._reshape(X)

        # Fit model
        clf = self._get_classifier()
        clf.fit(X_train, y['y_train'])

        preds_test = clf.predict(X_test)

        # Get evalutation score on test set
        eval_score = roc_auc_score(y['y_test'], preds_test)

        return eval_score

def main(argv):
    evaluator=SKLearnEvaluator(seed=FLAGS.seed, imputation=FLAGS.sklearn_imputation,
                               input_mode=FLAGS.sklearn_input_mode,
                               classifier=FLAGS.sklearn_model)
    evaluator.evaluate()

    eval_score = evaluator.evaluate()
    logging.info(F'Evaluation finished! AUROC: {eval_score}.')

    return eval_score

if __name__ == '__main__':
    app.run(main)