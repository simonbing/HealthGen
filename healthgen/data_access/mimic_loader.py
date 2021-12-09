"""
2021 Simon Bing, ETHZ, MPI IS
"""
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from absl import flags, app
from healthgen.data_access.base_loader import BaseLoader
from healthgen.data_access.preprocessing import Standardize, MissingnessDeltaT

FLAGS=flags.FLAGS

flags.DEFINE_string('input_vitals', '', 'Path to extracted patient vitals time series.')
flags.DEFINE_string('input_outcomes', '', 'Path to extracted patient outcomes time series.')
flags.DEFINE_string('input_static', '', 'Path to extracted patient static data.')
flags.DEFINE_integer('N_patients', None, 'Number of patients to consider. Pass None for all patients.')
flags.DEFINE_integer('time_resolution', 15, 'Time step size in minutes.')
flags.DEFINE_integer('time_length', 6, 'Duration of time series in hours.')
flags.DEFINE_integer('prediction_gap', 6, 'Time gap between observation and prediction in hours.')
flags.DEFINE_integer('prediction_window', 4, 'Time length of prediciton window in hours.')


class MimicLoader(BaseLoader):
    def __init__(self, seed, task_name, extracted_intermed_features_path,
                 extracted_intermed_labels_path, extracted_intermed_static_path,
                 processed_features_path, processed_labels_path,
                 vitals_path, outcomes_path, static_path, time_resolution, time_length,
                 prediction_gap, prediction_window, N_patients=None):
        super().__init__(seed, task_name, extracted_intermed_features_path,
                         extracted_intermed_labels_path, extracted_intermed_static_path,
                         processed_features_path, processed_labels_path)
        self.vitals_path = vitals_path
        self.outcomes_path = outcomes_path
        self.static_path = static_path

        self.N_patients = N_patients
        self.time_resolution = time_resolution
        self.time_length = time_length

        self.prediction_gap = prediction_gap
        self.prediction_window = prediction_window

        self.seed = seed

    def _get_icu_ids(self, df):
        """
        Get indices of icu admissions to keep, from pandas dataframe.
        Returns:
            icu_ids_slice: Indices of ids to keep.
            counts: Number of timesteps per id.
        """
        np.random.seed(self.seed)
        try:
            icu_ids, counts = np.unique(df.index.get_level_values('icustay_id').to_numpy(), return_counts=True)
        except KeyError:
            icu_ids, counts = np.unique(df['icustay_id'].to_numpy(), return_counts=True)
        N_icu_ids = len(icu_ids)

        if self.N_patients == N_icu_ids or self.N_patients is None:
            icu_ids_slice = icu_ids
            self.N_patients = N_icu_ids
        else:
            if self.N_patients < N_icu_ids:
                ids_slice = np.random.choice(N_icu_ids, self.N_patients, replace=False)
                icu_ids_slice = icu_ids[ids_slice]
                counts = counts[ids_slice]
            else:
                raise ValueError(
                    F'Selected more patients than available ({N_icu_ids}) in dataset!')

        return icu_ids_slice, counts


    def _compute_window_label(self, seq):
        """
        Compute the window label from a sequence of labels.
        Mapping: 0: wean on, 1: wean off, 2: stay on, 3:stay off

        Args:
            seq: 1D sequence of labels
        Returns:
            training label
        """
        if np.all(seq==0):
            return 3
        elif np.all(seq==1):
            return 2
        elif np.any(np.diff(seq)==-1):
            if seq[0]==1:
                return 1
            else:
                return 0
        elif np.any(np.diff(seq)==1):
            if seq[0]==0:
                return 0
            else:
                return 1


    def _compute_binary_label(self, x):
        """
        Compute binary label from window label.

        Args:
            x: array of window labels
        Returns:
            binary label for duration of entire window
        """
        if x==3:
            return 0
        else:
            return 1


    def get_timeseries(self):
        """
        Returns array containing raw time series per feature per patient.

        Returns:
            vitals_out: Array of all patients' vitals timeseries [N_patients, features, time_len]
        """
        # Load data
        vitals_raw = pd.read_hdf(self.vitals_path)

        # Get ids to keep
        icu_ids_slice, counts = self._get_icu_ids(vitals_raw)

        # TODO: move to seperate function!
        # Get all present features
        # all_features = list(vitals_raw.columns.levels[0])
        # if self.features_to_keep is None:
        #     features_slice = all_features
        # else:
        #     assert set(self.features_to_keep).issubset(all_features), 'Features not included in raw data!'
        #     features_slice = self.features_to_keep

        # Get extracted features
        feature_names = list(vitals_raw.columns.levels[0])

        # Get number of time steps to keep
        time_steps = int((60 / self.time_resolution) * self.time_length)
        assert time_steps <= min(counts), \
            F'Selected time_length too long, ' \
            F'please choose {(min(counts)*self.time_resolution)//60} hours or less!'

        # Drop unwanted patients, keep only mean value, truncate time length
        idx = pd.IndexSlice
        # vitals_truncated = vitals_raw.loc[idx[:,:,icu_ids_slice,:time_steps], idx[features_slice,'mean']]
        vitals_truncated = vitals_raw.loc[idx[:,:,icu_ids_slice,:time_steps], idx[:,'mean']]

        vitals_np = vitals_truncated.to_numpy()
        # vitals_reshaped = np.reshape(vitals_np, (len(icu_ids_slice), time_steps+1, len(features_slice)))
        vitals_reshaped = np.reshape(vitals_np, (len(icu_ids_slice), time_steps+1, -1))
        vitals_out = np.transpose(vitals_reshaped, (0,2,1))

        return vitals_out, np.asarray(feature_names)


    def get_outcomes(self):
        """
        Returns array containing labels time series per target per patient.

        Returns:
            outcomes_out: Array of all patients' labels [N_patients, targets]
        """
        # Load data
        outcomes_raw = pd.read_hdf(self.outcomes_path)

        # Get ids to keep
        icu_ids_slice, counts = self._get_icu_ids(outcomes_raw)

        # Get steps corresponding to prediction window
        window_start = int((60 / self.time_resolution) * (self.time_length + self.prediction_gap))
        window_end = window_start + int((60 / self.time_resolution) * self.prediction_window)
        assert window_end <= min(counts), F'Selected prediction_gap + prediction_window too long!'

        # Drop time patients and time steps we don't want
        idx = pd.IndexSlice
        logging.info(outcomes_raw.columns)
        outcomes_truncated = outcomes_raw.loc[idx[:,:,icu_ids_slice,window_start+1:window_end], idx[:]]

        outcomes_np = outcomes_truncated.to_numpy()
        outcomes_reshaped = np.reshape(outcomes_np, (len(icu_ids_slice), window_end-window_start, outcomes_np.shape[1]))

        # Compute labels for window (non-binary) tasks (all but idxs 8,9)
        outcomes_window = np.apply_along_axis(self._compute_window_label, 1, outcomes_reshaped)
        vectorized_compute_binary_label = np.vectorize(self._compute_binary_label)
        outcomes_binary = vectorized_compute_binary_label(outcomes_window)

        # Concatenate all labels, window and binary
        # TODO: USE THIS ONE AGAIN FOR CLUSTER!
        window_mask = np.array([1,1,1,1,1,1,1,1,1,1,1,0,0,1], dtype=bool)
        # window_mask = np.array([1,1,1,1,1,1,1,1,0,0,1], dtype=bool)

        outcomes_out = np.concatenate((outcomes_binary, outcomes_window[:, window_mask]), axis=1)

        return outcomes_out


    def get_static_data(self):
        """
        Returns array containing static data for each patient.

        Returns:
            static_data: [N_patients, static_features]
            feature_names: [static_features]
        """
        # Load data
        static_raw = pd.read_csv(self.static_path)

        # Get ids to keep
        icu_ids_slice, counts = self._get_icu_ids(static_raw)

        # Static features to keep, hardcoded for now
        feats_to_keep = ['gender', 'ethnicity', 'age', 'insurance', 'admittime',
                         'dischtime', 'diagnosis_at_admission', 'intime', 'outtime',
                         'admission_type', 'first_careunit', 'readmission_30']

        # Truncate features we do not wish to keep
        idx = pd.IndexSlice
        static_raw = static_raw.set_index('icustay_id')
        static_truncated = static_raw.loc[idx[icu_ids_slice], idx[feats_to_keep]]

        # Map features to integer encoding
        def ethnicity_map(item):
            if 'WHITE' in item:
                return 0
            elif 'BLACK' in item:
                return 1
            elif 'HISPANIC' in item:
                return 2
            elif 'ASIAN' in item:
                return 3
            else:
                return 4

        def insurance_map(item):
            if 'Medicare' in item:
                return 0
            elif 'Medicaid' in item:
                return 1
            elif 'Private' in item:
                return 2
            elif 'Government' in item:
                return 3
            elif 'Self' in item:
                return 4
            else:
                raise ValueError

        def admission_type_map(item):
            if 'EMERGENCY' in item:
                return 0
            elif 'ELECTIVE' in item:
                return 1
            elif 'URGENT' in item:
                return 2
            else:
                raise ValueError

        def first_careunit_map(item):
            if 'MICU' in item:
                return 0
            elif 'CSRU' in item:
                return 1
            elif 'SICU' in item:
                return 2
            elif 'CCU' in item:
                return 3
            elif 'TSICU' in item:
                return 4
            else:
                raise ValueError

        static_truncated['gender'] = [0 if i == 'F' else 1 for i in static_truncated['gender']]
        static_truncated['ethnicity'] = list(map(ethnicity_map, static_truncated['ethnicity']))
        static_truncated['age'] = [int(i) if i < 89 else 89 for i in static_truncated['age']]
        static_truncated['insurance'] = list(map(insurance_map, static_truncated['insurance']))
        static_truncated['admittime'] = [datetime.timestamp(datetime.strptime(i, "%Y-%m-%d %H:%M:%S")) for i in static_truncated['admittime']]
        static_truncated['dischtime'] = [datetime.timestamp(datetime.strptime(i, "%Y-%m-%d %H:%M:%S")) for i in static_truncated['dischtime']]
        static_truncated['diagnosis_at_admission'] = [0 for i in static_truncated['diagnosis_at_admission']] # TODO: find sensible mapping
        static_truncated['intime'] = [datetime.timestamp(datetime.strptime(i, "%Y-%m-%d %H:%M:%S")) for i in static_truncated['intime']]
        static_truncated['outtime'] = [datetime.timestamp(datetime.strptime(i, "%Y-%m-%d %H:%M:%S")) for i in static_truncated['outtime']]
        static_truncated['admission_type'] = list(map(admission_type_map, static_truncated['admission_type']))
        static_truncated['first_careunit'] = list(map(first_careunit_map, static_truncated['first_careunit']))

        static_out = {'c': static_truncated.to_numpy(),
                      'feature_names': np.asarray(feats_to_keep)}

        return static_out


    def select_features(self, X, features):
        """
        Args:
            X: dict containing at least 'inputs': [N_patients, features, time_len]
               and 'feature_names': [features]
            features: list of features to keep

        Returns:
            X: dict with entries containing only the desired features
        """
        if features is None:
            return X
        else:
            # feature_idxs_to_keep = np.where(np.isin(X['feature_names'], features)).tolist()
            feature_mask = np.isin(X['feature_names'], features)
            X_tf = {}
            for key, value in X.items():
                try:
                    X_tf[key] = value[:,feature_mask,...]
                except IndexError: # Handle feature_names
                    X_tf[key] = value[feature_mask]
            return X_tf


    def select_task(self, outcomes, task):
        """
        Returns associated labels of the specified task.

        Args:
            outcomes: all available outcomes
            task: name of the chosen task
        """
        if task is None:
            return outcomes
        else:
            mimic_task_map = ['vent_bin', 'vaso_bin', 'adenosine_bin', 'dobutamine_bin',
                              'dopamine_bin', 'epinephrine_bin', 'isuprel_bin', 'milrinone_bin',
                              'norepinephrine_bin', 'phenylephrine_bin', 'vasopressin_bin',
                              'colloid_bolus_bin', 'crystalloid_bolus_bin', 'nivdurations_bin',
                              'vent_win', 'vaso_win', 'adenosine_win','dopamine_win', 'dobutamine_win',
                              'epinephrine_win', 'isuprel_win', 'milrinone_win', 'norepinephrine_win',
                              'phenylephrine_win', 'vasopressin_win', 'nivdurations_win']

            return outcomes[:, mimic_task_map.index(task)]


    def get_intermed_preprocessors(self):
        preprocessors = [
            MissingnessDeltaT()
        ]

        return preprocessors

    def get_input_preprocessors(self):
        preprocessors = [
            Standardize(),
        ]

        return preprocessors

    def save_intermed_data(self, X, y, c, out_path):
        np.savez(os.path.join(out_path,
                              F'X_{self.time_length}hrs_{self.time_resolution}min_'
                              F'{self.N_patients}_intermed'),
                 X=X['X'], m=X['m'], delta_t=X['delta_t'],
                 feature_names=X['feature_names'])
        np.save(os.path.join(out_path,
                             F'y_{self.time_length}hrs_{self.time_resolution}min_'
                             F'{self.N_patients}_intermed'), y)
        np.savez(os.path.join(out_path,
                             F'c_{self.time_length}hrs_{self.time_resolution}min_'
                             F'{self.N_patients}_intermed.npz'), **c)

    def save_processed_data(self, X, y, out_path):
        np.savez(os.path.join(out_path,
                              F'X_{self.time_length}hrs_{self.time_resolution}min_'
                              F'{self.N_patients}_{self.task_name}_proc'),
                 X_train=X['X_train'], X_val=X['X_val'], X_test=X['X_test'],
                 m_train=X['m_train'], m_val=X['m_val'], m_test=X['m_test'],
                 delta_t_train=X['delta_t_train'], delta_t_val=X['delta_t_val'],
                 delta_t_test=X['delta_t_test'], feature_names=X['feature_names'])
        np.savez(os.path.join(out_path,
                              F'y_{self.time_length}hrs_{self.time_resolution}min_'
                              F'{self.N_patients}_{self.task_name}_proc'), **y)
        # np.savez(os.path.join(out_path,
        #                       F'c_{self.time_length}hrs_{self.time_resolution}min_'
        #                       F'{self.N_patients}_{self.task_name}_proc'),
        #          c_train=c['c_train'], c_val=c['c_val'], c_test=c['c_test'],
        #          feature_names=c['feature_names'])


def main(argv):
    loader = MimicLoader(seed=FLAGS.seed, task_name=FLAGS.task,
                         processed_features_path=FLAGS.processed_features_path,
                         processed_labels_path=FLAGS.processed_labels_path,
                         extracted_intermed_features_path=FLAGS.extracted_intermed_features_path,
                         extracted_intermed_labels_path=FLAGS.extracted_intermed_labels_path,
                         vitals_path=FLAGS.input_vitals, outcomes_path=FLAGS.input_outcomes,
                         static_path=FLAGS.input_static,
                         N_patients=FLAGS.N_patients, time_resolution=FLAGS.time_resolution,
                         time_length=FLAGS.time_length, prediction_gap=FLAGS.prediction_gap,
                         prediction_window=FLAGS.prediction_window)
    X, y = loader.get_data()
    print('Loaded data!')
    # if FLAGS.save_processed_data:
    #     loader.save_processed_data(X, y, FLAGS.processed_output_path)
    #     print('Saved data!')

if __name__ == '__main__':
    app.run(main)