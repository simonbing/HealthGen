"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'], 'size':15})


def plot_reconstructions(orig_data, recon_data, patient_idx, data_mode='feats_mask', feature_names=None,
                         features_to_plot=None):
    """
    Plot reconstructions of time series data.

    Args:
        orig_data: original time series data array [N, features, time_len]
        recon_data: reconstructed time series data array [N, features, time_len]
        feature_names: list of named features present in the data.
        features_to_plot: list of names of features which we wish to plot.
    """
    # Keep only features which we want to plot
    # Plot maximum of 5 features, use first features for now
    tot_num_features = orig_data.shape[-2]
    num_features_to_plot = np.min((tot_num_features, 5))
    if data_mode == 'feats'or data_mode == 'all':
        features_idxs = np.arange(num_features_to_plot)
    else:
        if data_mode == 'feats_mask':
            masks = orig_data[patient_idx, 1, :, :]
        elif data_mode == 'mask':
            masks = orig_data[patient_idx, :, :]
        # Get features with least missingness
        miss_rate = np.sum(masks, axis=1)
        features_idxs = miss_rate.argpartition(-num_features_to_plot)[-num_features_to_plot:]

    orig_data = orig_data[..., features_idxs, :]
    recon_data = recon_data[..., features_idxs, :]

    # Plot
    colors = plt.rcParams["axes.prop_cycle"]()

    if data_mode == 'feats_mask':
        fig, axs = plt.subplots(nrows=num_features_to_plot * 2, ncols=2,
                                figsize=(30, 3.5 * num_features_to_plot * 2))
        for idx in range(num_features_to_plot):
            c = next(colors)["color"]

            # feats
            axs[idx, 0].plot(orig_data[patient_idx, 0, idx, :], color=c)
            axs[idx, 1].plot(recon_data[patient_idx, 0, idx, :], color=c)
            # masks
            axs[idx+num_features_to_plot, 0].plot(orig_data[patient_idx, 1, idx, :], color=c)
            axs[idx+num_features_to_plot, 1].plot(recon_data[patient_idx, 1, idx, :], color=c)

            axs[0, 0].set_title('Original')
            axs[0, 1].set_title('Reconstruction')
    else:
        fig, axs = plt.subplots(nrows=num_features_to_plot, ncols=2,
                                figsize=(30, 3.5 * num_features_to_plot))
        for idx in features_idxs:
            c = next(colors)["color"]

            try:
                axs[idx, 0].plot(orig_data[patient_idx, idx, :], color=c)
                # axs[idx, 0].set_ylabel(feature)
                axs[idx, 1].plot(recon_data[patient_idx, idx, :], color=c)
                axs[0, 0].set_title('Original')
                axs[0, 1].set_title('Reconstruction')
            except IndexError:
                axs[0].plot(orig_data[patient_idx, idx, :], color=c)
                # axs[0].set_ylabel(feature)
                axs[1].plot(recon_data[patient_idx, idx, :], color=c)
                axs[0].set_title('Original')
                axs[1].set_title('Reconstruction')

    # fig.show()
    return fig

def get_synth_labels(y_real, mode='synth', split=None, cond_static=False, static_vars=None):
    """
    Args:
        y_real: Dictionary of all labels (and static variables).
        mode: Whether to augment real data or synthesize entire new set.
        split: Class split of labels. Returns original labels for None.
        cond_static: Whether or not to condition on additional static variables.
        static_vars: Static variables to condition on.

    Returns:
        labels: All labels on which to condition generation.
    """
    labels_all = np.concatenate((y_real['y_train'], y_real['y_val'], y_real['y_test']))
    if cond_static: # Include static variables for conditioning
        if split is None:
            if static_vars is None:
                static_vars_idxs = np.arange(len(y_real['feature_names']))
            else:
                static_vars_idxs = np.flatnonzero(np.in1d(y_real['feature_names'], static_vars))
            static_all = np.concatenate((y_real['c_train'], y_real['c_val'], y_real['c_test']))[:,static_vars_idxs].squeeze()
            if 'age' in static_vars:
                static_all = [0 if value <= 30 else 1 if value <= 50 else 2 if value <= 70 else 3 for value in static_all]
            labels_all = np.stack((labels_all, static_all), axis=1)

            return labels_all
        else:
            if mode == 'synth':
                if 'gender' in static_vars:
                    static_all = np.random.choice([0, 1], size=len(labels_all))
                elif 'age' in static_vars:
                    static_all = np.random.choice([0, 1, 2, 3], size=len(labels_all))
                elif 'ethnicity' in static_vars:
                    static_all = np.random.choice([0, 1, 2, 3, 4], size=len(labels_all))
                elif 'insurance' in static_vars:
                    static_all = np.random.choice([0, 1, 2, 3, 4], size=len(labels_all))
            elif mode == 'augment':
                if static_vars is None:
                    static_vars_idxs = np.arange(len(y_real['feature_names']))
                else:
                    static_vars_idxs = np.flatnonzero(np.in1d(y_real['feature_names'], static_vars))
                static_all_orig = np.concatenate((y_real['c_train'], y_real['c_val'], y_real['c_test']))[:, static_vars_idxs].squeeze()

                if 'gender' in static_vars:
                    genders, gender_counts = np.unique(static_all_orig, return_counts=True)
                    majority_count = gender_counts[-1]
                    static_all = np.zeros(int(majority_count-gender_counts[0]))
                elif 'age' in static_vars:
                    static_all_orig = [0 if value <= 30 else 1 if value <= 50 else 2 if value <= 70 else 3 for value in static_all_orig]
                    ages, age_counts = np.unique(static_all_orig, return_counts=True)
                    majority_count = age_counts[-1]
                    sub_30 = np.zeros(int(majority_count-age_counts[0]))
                    sub_50 = np.ones(int(majority_count-age_counts[1]))
                    sub_70 = 2 * np.ones(int(majority_count - age_counts[2]))
                    static_all = np.concatenate((sub_30, sub_50, sub_70))
                elif 'ethnicity' in static_vars:
                    ethnicities, ethnicity_counts = np.unique(static_all_orig, return_counts=True)
                    majority_count = ethnicity_counts[0]
                    black = np.ones(majority_count-ethnicity_counts[1])
                    hisp = 2 * np.ones(majority_count - ethnicity_counts[2])
                    asian = 3 * np.ones(majority_count - ethnicity_counts[3])
                    other = 4 * np.ones(majority_count - ethnicity_counts[4])
                    static_all = np.concatenate((black, hisp, asian, other))
                elif 'insurance' in static_vars:
                    insurances, insurance_counts = np.unique(static_all_orig, return_counts=True)
                    majority_count = insurance_counts[0]
                    medicaid = np.ones(majority_count - insurance_counts[1])
                    priv = 2 * np.ones(majority_count - insurance_counts[2])
                    gov = 3 * np.ones(majority_count - insurance_counts[3])
                    self_pay = 4 * np.ones(majority_count - insurance_counts[4])
                    static_all = np.concatenate((medicaid, priv, gov, self_pay))
                curr_split = np.sum(labels_all) / len(labels_all)
                labels_all = np.random.choice([0., 1.], size=len(static_all), p=[1-curr_split, curr_split])
            else:
                raise ValueError

            labels_all = np.stack((labels_all, static_all), axis=1)

            return labels_all

    else: # Only labels:
        if split is None:
            labels = labels_all
            return labels
        else:
            if mode == 'synth':
                labels = np.random.choice([0., 1.], size=len(labels_all), p=[1-split, split] )
                return labels
            elif mode == 'augment':
                labels_train_val = np.concatenate((y_real['y_train'], y_real['y_val']))
                tot_num_samples = len(labels_all)

                curr_split = np.sum(labels_train_val) / len(labels_train_val)
                p_have = int(curr_split * tot_num_samples)
                n_have = tot_num_samples - p_have

                p_add = int((split/(1-split)) * n_have - p_have)
                labels = np.ones(p_add)

                ### Hack for per class robustness eval of baselines ###
                # curr_split = np.sum(labels_train_val) / len(labels_train_val)
                # p_add = 57113
                # labels = np.random.choice([0., 1.], size=p_add, p=[1-curr_split, curr_split])
                #######################################################

                return labels
            else:
                raise ValueError

def augment_data(X_real, y_real, X_synth, y_synth):
    np.random.seed(0) # To ensure shuffling is reproducible

    # Only take actual label if we also used static variables for conditioning
    if len(y_synth['y_train'].shape) > 1:
        y_synth['y_train'] = y_synth['y_train'][:,0]
        y_synth['y_val'] = y_synth['y_val'][:, 0]
        y_synth['y_test'] = y_synth['y_test'][:, 0]

    # Concatenate all subsets
    X_train = np.concatenate((X_real['X_train'], X_synth['X_train']))
    X_val = np.concatenate((X_real['X_val'], X_synth['X_val']))
    X_test = np.concatenate((X_real['X_test'], X_synth['X_test']))

    m_train = np.concatenate((X_real['m_train'], X_synth['m_train']))
    m_val = np.concatenate((X_real['m_val'], X_synth['m_val']))
    m_test = np.concatenate((X_real['m_test'], X_synth['m_test']))

    delta_t_train = np.concatenate((X_real['delta_t_train'], X_synth['delta_t_train']))
    delta_t_val = np.concatenate((X_real['delta_t_val'], X_synth['delta_t_val']))
    delta_t_test = np.concatenate((X_real['delta_t_test'], X_synth['delta_t_test']))

    y_train = np.concatenate((y_real['y_train'], y_synth['y_train']))
    y_val = np.concatenate((y_real['y_val'], y_synth['y_val']))
    y_test = np.concatenate((y_real['y_test'], y_synth['y_test']))

    # Get shuffling permutation
    train_perm = np.arange(len(X_train))
    val_perm = np.arange(len(X_val))
    test_perm = np.arange(len(X_test))
    np.random.shuffle(train_perm)
    np.random.shuffle(val_perm)
    np.random.shuffle(test_perm)

    # Permute all arrays with with same vector
    X_train = X_train[train_perm, ...]
    m_train = m_train[train_perm, ...]
    delta_t_train = delta_t_train[train_perm, ...]
    y_train = y_train[train_perm]

    X_val = X_val[val_perm, ...]
    m_val = m_val[val_perm, ...]
    delta_t_val = delta_t_val[val_perm, ...]
    y_val = y_val[val_perm]

    X_test = X_test[test_perm, ...]
    m_test = m_test[test_perm, ...]
    delta_t_test = delta_t_test[test_perm, ...]
    y_test = y_test[test_perm]

    X_aug = {'X_train': X_train,
             'X_val': X_val,
             'X_test': X_test,
             'm_train': m_train,
             'm_val': m_val,
             'm_test': m_test,
             'delta_t_train': delta_t_train,
             'delta_t_val': delta_t_val,
             'delta_t_test': delta_t_test}

    y_aug = {'y_train': y_train, 'y_val': y_val, 'y_test': y_test}

    return X_aug, y_aug

def split_labels_per_class(x, y, c, stat_feat_names, stat_feat='ethnicity'):
    """
    Args:
        x: Features, array [N, ...]
        y: Labels for classification, array [N]
        c: Static features, array [N, M]
        stat_feat_names: List of static feature names, [m]
        stat_feat: Name of static feature by which to split labels, string

    Returns:
        x_split_list, y_split_list
    """
    stat_feat_idx = np.argwhere(stat_feat_names==stat_feat)[0,0]

    # If we are dealing with age, first bin age groups
    if stat_feat == 'age':
        c[:, stat_feat_idx] = [0 if value <= 30 else 1 if value <= 50 \
                               else 2 if value <= 70 else 3 for value in c[:, stat_feat_idx]]

    # Get indices of individual classes
    unique_values = np.unique(c[:, stat_feat_idx])
    # print(F'Unique labels: {unique_values}')
    value_idxs = []
    y_split_list = []
    x_split_list = []

    for value in unique_values:
        idxs = np.argwhere(c[:, stat_feat_idx]==value).squeeze()
        value_idxs.append(idxs)
        y_split_list.append(y[idxs])
        x_split_list.append(x[idxs, ...])

    return x_split_list, y_split_list

# vent, vaso, colloid_bolus, crystalloid, nivdurations
