###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

from scivae import VAE, VAEStats
from scipy import stats
import numpy as np
import pandas as pd
from sciutil import SciUtil
import json


class RCMVAE:
    """
    Enables running a VAE on the RCM data.
    Takes in the output of rcm and then for each SiRCLe cluster, trains a VAE. It returns the dataset which can be
    used as input to calculating statistics on the DS.
    """

    def __init__(self, rcm_file: str,
                 meth_file: str, meth_sample_file: str,
                 rna_file: str, rna_sample_file: str,
                 protein_file: str, protein_sample_file: str,
                 output_folder: str, stat_condition_column: str,  # This is a a column that must be present in all sample dfs
                 sub_condition_id_column: str,
                 # For example stage of the patient.
                 config_file: str=None, regulatory_label='Regulation_Grouping_2',
                 run_name: str = None, normalise = 'columns'):
        self.config_json = f'{config_file}'
        self.feature_columns = ['RNA-LogFC',
                                'Protein-LogFC',
                                'CpG-LogFC',
                                'RNA-Tumor',
                                'RNA-Normal',
                                'Protein-Tumor',
                                'Protein-Normal']
        # Read in the data
        self.regulatory_label = regulatory_label
        self.rcm = pd.read_csv(rcm_file, index_col=0)
        self.meth_data = pd.read_csv(meth_file, index_col=0)
        self.rna_data = pd.read_csv(rna_file, index_col=0)
        self.protein_data = pd.read_csv(protein_file, index_col=0)
        self.run_name = run_name
        # Read in the sample dfs
        self.meth_sample_df = pd.read_csv(meth_sample_file)
        self.rna_sample_df = pd.read_csv(rna_sample_file)
        self.protein_sample_df = pd.read_csv(protein_sample_file)
        self.stat_condition_column = stat_condition_column
        self.stat_condition_id_column = sub_condition_id_column
        # For each of these files, normalise the rows to be between 0 and 1.
        self.output_folder = output_folder
        self.normalise = normalise
        self.sample_df = pd.DataFrame(columns=['case_id', 'condition_id', 'column_label', 'column_id', 'multi_loss'])
        self.u = SciUtil()

    def train_vae(self, config=None):
        """
        Sets up and normalises the data.
        :return:
        """
        for reg_label in self.rcm[self.regulatory_label].unique():
            if reg_label != "None":
                rcm_df = self.rcm[self.rcm[self.regulatory_label] == reg_label].copy()
                meth_data = self.compute_columns(self.align_to_rcm(self.meth_data, rcm_df), self.meth_sample_df,
                                                 'CpG', include_missing=False)
                rna_data = self.compute_columns(self.align_to_rcm(self.rna_data, rcm_df), self.rna_sample_df, 'RNA',
                                                include_missing=False)
                protein_data = self.compute_columns(self.align_to_rcm(self.protein_data, rcm_df,
                                                                      include_regulatory_label=True),
                                                    self.protein_sample_df, 'Protein', include_missing=False)
                r_df = pd.concat(
                    [protein_data,
                     rna_data,
                     meth_data], axis=1)
                # Create a sample df based on the columns
                train_df = self.build_training_df(r_df)
                self.train(train_df, self.feature_columns, reg_label, config)

    def train(self, train_df, feature_columns, reg_label, config):
        epochs = 100
        batch_size = 16
        if config is None:
            loss = {'loss_type': 'multi', 'distance_metric': 'mmd', 'mmd_weight': 0.1, 'multi_loss': ['mse', 'mse']}

            config = {"loss": loss,
                      "encoding": {"layers": [[{"num_nodes": 16, "activation_fn": "relu"},
                                               {"num_nodes": 16, "activation_fn": "relu"}]]},
                      "decoding": {"layers": [[{"num_nodes": 16, "activation_fn": "relu"},
                                               {"num_nodes": 16, "activation_fn": "relu"}]]},
                      "latent": {"num_nodes": 1},
                      "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                      "input_size": [3, 4],
                      "output_size": [3, 4],
                      "epochs": epochs,
                      "batch_size": batch_size
            }
        else:
            epochs = config.get('epochs', epochs)
            batch_size = config.get('batch_size', batch_size)

        sub_sample = train_df
        if config.get('loss').get('loss_type') == 'multi':
            data_values = [sub_sample[['RNA-LogFC', 'Protein-LogFC', 'CpG-LogFC']].values,
                           sub_sample[['RNA-Tumor', 'RNA-Normal', 'Protein-Tumor', 'Protein-Normal']].values]
        else:
            data_values = sub_sample[feature_columns].values
        labels = list(sub_sample.index.values)
        vae_m = VAE(data_values, data_values, labels, config, vae_label=reg_label)
        # epochs=50, batch_size=50, train_percent=85.0, logging_dir=None, logfile=None
        vae_m.encode('default', train_percent=75.0, epochs=epochs, batch_size=batch_size,
                     logging_dir=self.output_folder,
                     logfile=f'VAE-logfile-{reg_label}-{self.run_name}.txt', early_stop=True)
        vae_m.save(weight_file_path=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-weights.h5',
                   optimizer_file_path=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-optimizer.json',
                   config_json=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-config.json')  # save the VAE
        vae_m.u.dp(["Saved VAE to current directory."])

    def get_decoding_for_cluster(self, reg_label, cases):
        weight_file_path = f'{self.output_folder}{reg_label}-{self.run_name}-VAE-weights.h5'
        optimizer_file_path = f'{self.output_folder}{reg_label}-{self.run_name}-VAE-optimizer.json'
        config_json = f'{self.output_folder}{reg_label}-{self.run_name}-VAE-config.json'
        rcm_df = self.rcm[self.rcm[self.regulatory_label] == reg_label].copy()
        meth_data = self.compute_columns(self.align_to_rcm(self.meth_data, rcm_df), self.meth_sample_df,
                                         'CpG', include_missing=False)
        rna_data = self.compute_columns(self.align_to_rcm(self.rna_data, rcm_df), self.rna_sample_df, 'RNA',
                                        include_missing=False)
        protein_data = self.compute_columns(self.align_to_rcm(self.protein_data, rcm_df,
                                                              include_regulatory_label=True),
                                            self.protein_sample_df, 'Protein', include_missing=False)
        r_df = pd.concat(
            [protein_data,
             rna_data,
             meth_data], axis=1)

        # Create a sample df based on the columns
        sample_df = self.build_stats_sample_df(r_df)
        train_df = self.build_training_df(r_df, False, cases)

        data_values = train_df[self.feature_columns].values
        labels = list(sample_df.index.values)
        config = {}
        with open(config_json, "r") as fp:
            config = json.load(fp)
        vae_m = VAE(data_values, data_values, labels, config, vae_label=reg_label)
        # Then decode the data
        vae_m.load(weight_file_path, optimizer_file_path, config_json)
        encoding = vae_m.encode_new_data(data_values, scale=False)
        decoding = vae_m.decoder.predict(encoding)
        enc_df = pd.DataFrame(data=encoding, columns=['VAE'])
        enc_df['id'] = list(train_df.index.values)
        dec_df = pd.DataFrame(data=decoding, columns=self.feature_columns)
        dec_df['id'] = list(train_df.index.values)
        train_df['id'] = list(train_df.index.values)
        return enc_df, dec_df, train_df

    def align_to_rcm(self, df, rcm_df, include_regulatory_label=False):
        aligned_df = pd.DataFrame()
        aligned_df['genes'] = rcm_df.index
        aligned_df.set_index('genes', inplace=True)
        aligned_df = aligned_df.join(df, how='left')  # Get all the data aligned to the same index
        aligned_df.fillna(0, inplace=True)  # Fill in the missing values with 0
        # normalise df
        normalised_aligned_df = self.normalise_df(aligned_df) # normalise the data
        if include_regulatory_label:
            normalised_aligned_df[self.regulatory_label] = rcm_df[self.regulatory_label].values

        return normalised_aligned_df

    def run_vae_stats(self, label='', include_missing=False, meth_sample_file = None, rna_sample_file = None,
                      protein_sample_file = None):
        # Enable a user to load in different sample files for different runs.
        if meth_sample_file is not None:
            self.meth_sample_df = pd.read_csv(meth_sample_file)
        if rna_sample_file is not None:
            self.rna_sample_df = pd.read_csv(rna_sample_file)
        if protein_sample_file is not None:
            self.protein_sample_df = pd.read_csv(protein_sample_file)

        # Now they all have the same index and are aligned so we can just build our training data by concating them
        for reg_label in self.rcm[self.regulatory_label].unique():
            if reg_label != "None":
                rcm_df = self.rcm[self.rcm[self.regulatory_label] == reg_label].copy()
                meth_data = self.compute_columns(self.align_to_rcm(self.meth_data, rcm_df), self.meth_sample_df,
                                                 'CpG', include_missing=False)
                rna_data = self.compute_columns(self.align_to_rcm(self.rna_data, rcm_df), self.rna_sample_df, 'RNA',
                                                include_missing=include_missing)
                protein_data = self.compute_columns(self.align_to_rcm(self.protein_data, rcm_df,
                                                                      include_regulatory_label=True),
                                                    self.protein_sample_df, 'Protein', include_missing=include_missing)
                r_df = pd.concat(
                    [protein_data,
                     rna_data,
                     meth_data], axis=1)

                # Create a sample df based on the columns
                sample_df = self.build_stats_sample_df(r_df)
                matched_cases = []
                for case in sample_df['case_id'].unique():
                    case_sample_df = sample_df[sample_df['case_id'] == case]
                    if len(case_sample_df) > 8:
                        matched_cases.append(case)
                    else:
                        continue  # print(case)
                sample_df = sample_df[sample_df['case_id'].isin(matched_cases)]
                self.u.err_p([reg_label])
                vs = VAEStats(r_df, sample_df, weight_file_path=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-weights.h5',
                              optimizer_file_path=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-optimizer.json',
                              config_json=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-config.json',
                              feature_columns=self.feature_columns)
                # Load config so we get the loss type
                fp = open(f'{self.output_folder}{reg_label}-{self.run_name}-VAE-config.json', 'r')
                config = json.load(fp)
                fp.close()
                if config.get('loss').get('loss_type') == 'multi':
                    stats_vals = vs.peform_DVAE_multiloss([['RNA-LogFC', 'Protein-LogFC', 'CpG-LogFC'],
                                                          ['RNA-Tumor', 'RNA-Normal', 'Protein-Tumor', 'Protein-Normal']],
                                                          column_to_align_to=['Protein-LogFC', 'RNA-LogFC', 'CpG-LogFC'])
                else:
                    stats_vals = vs.peform_DVAE(column_to_align_to=['Protein-LogFC', 'RNA-LogFC', 'CpG-LogFC'])

                vs.test_for_normality(stats_vals['base_mean_cond_0'])
                vs.test_for_normality(stats_vals['base_mean_cond_1'])
                stats_vals.to_csv(f'{self.output_folder}{reg_label}-{self.run_name}-stats{label}.csv')

    def build_stats_sample_df(self, df):
        sample_df = pd.DataFrame(columns=['case_id', 'condition_id', 'column_label', 'column_id', 'multi_loss'])
        # ['case_id', 'condition_id', 'column_label', 'column_id', 'multi_loss'] sample df columns
        case_ids = []
        condition_ids = []
        column_label = []
        column_id = []
        multi_loss = []
        for c in df.columns:
            if c != 'id' and c != self.regulatory_label:
                info = c.split('_')
                # # [f'{sub_condition}_{stats_cond_map}_{case}_{label}-Normal']
                case_ids.append(info[2])
                # ToDO Warn here
                condition_ids.append(int(info[1]))
                if 'LogFC' in c:
                    multi_loss.append(0)
                else:
                    multi_loss.append(1)
                column_id.append(c)
                column_label.append(info[-1])
        sample_df['case_id'] = case_ids
        sample_df['condition_id'] = condition_ids
        sample_df['column_label'] = column_label
        sample_df['column_id'] = column_id
        sample_df['multi_loss'] = multi_loss
        return sample_df

    def build_training_df(self, df, filter_extremes=True, selected_cases=None):
        # We want to add all the case data as training data and just keep the columns in the correct order
        # Basically we do this for all cases.
        cases = list(set(['_'.join(c.split('_')[:-1]) for c in df.columns if c != 'id']))  # ToDO: CHECK!!!!!
        train_df = pd.DataFrame()
        if selected_cases is not None:
            cases_subset = [c for c in cases if c.split('_')[-1] in selected_cases]
            cases = cases_subset
        for case in cases:
            case_cond_df = pd.DataFrame()
            case_cond_df['id'] = list(df.index.values)
            for col in self.feature_columns:
                try:
                    case_cond_df[col] = df[f'{case}_{col}'].values  # Get the column name from the case
                except:
                    continue  # self.u.warn_p([f'{case}_{col} not in df... Skipping...'])
            # Add this to the cond_1_sample_df
            if len(case_cond_df.columns) == len(self.feature_columns) + 1:  # For the index column
                train_df = train_df.append(case_cond_df)
        train_df.set_index('id', inplace=True)
        if filter_extremes:
            z_score = np.abs(stats.zscore(train_df, axis=1))
            max_z_score = np.max(z_score, axis=1)
            train_df = train_df[max_z_score < 2]
        return train_df

    def normalise_df(self, df):
        """
        Formats the dataframe.
        :return:
        """
        # Next normalise
        # We want to add all the case data as training data and just keep the columns in the correct order
        # Basically we do this for all cases.

        if self.normalise == 'rows':
            data = df.values.copy()
            # Min max scale this data
            data_values = []  # Basically just going to normalise each row
            for i, row in enumerate(data):
                if row.max() == 0:
                    data_values.append(row)
                else:
                    non_zero_min = np.min(row[row > 0])
                    non_zero_max = np.max(row[row > 0])
                    new_values = []
                    for j, val in enumerate(row):
                        if val > 0:
                            new_values.append((val - non_zero_min) / (non_zero_max - non_zero_min))
                        else:
                            new_values.append(0)
                    data_values.append(new_values)

            # Refill in the normalised protein data
            new_df = pd.DataFrame(data_values, index=df.index, columns=df.columns)
            return new_df
        elif self.normalise == 'columns':
            scaled_df = pd.DataFrame()
            scaled_df['genes'] = df.index.values
            numeric_cols = [c for c in df.columns if c != self.regulatory_label and c != 'id']
            # For each column, normalise to min-max but fist ommit any 0's
            for col in numeric_cols:
                values = df[col].values.copy()
                if values.max() == 0:
                    scaled_df[col] = values
                else:
                    non_zero_values = values[values != 0]
                    min_nz = np.min(non_zero_values)
                    max_nz = np.max(non_zero_values)
                    values[values != 0] = (values[values != 0] - min_nz) / (max_nz - min_nz)  # Min max scale and
                    # leave the rest 0's
                    scaled_df[col] = values
            scaled_df.set_index('genes', inplace=True)
            return scaled_df

    def compute_columns(self, df, sample_df, label, include_missing=False):
        # Leave does that had complete 0's as complete 0's
        # Next we want to build the features for the model (namely, the case wise change in each data type).
        vae_data_df = pd.DataFrame()
        missing_cond_0_data, missing_cond_1_data = 0, 0
        count_comparisons = 0

        vae_data_df['id'] = list(df.index.values)  # The index must be the gene ID
        # First split by the sub condition (we want to keep the data separate).
        # Get the stats condition map
        stats_cond_map = dict(zip(sample_df[self.stat_condition_column].values, sample_df[self.stat_condition_id_column].values))
        for sub_condition in sample_df[self.stat_condition_column].unique():
            # Get the mean of all normal samples
            sub_sample_df = sample_df[sample_df[self.stat_condition_column] == sub_condition]
            cond_0_columns_all = sub_sample_df[sub_sample_df['condition_id'] == 0]['column_id'].values
            # We use these if either value is missing.
            if len(cond_0_columns_all) == 0:
                continue  #print(sub_condition)
            cond_0_mean = np.mean(df[cond_0_columns_all].values, axis=1)

            for case in sub_sample_df['case_id'].unique():
                # Want to do condition 1 - condition 0 --> keeping with standard approach
                case_df = sub_sample_df[sub_sample_df['case_id'] == case]
                cond_0_columns = case_df[case_df['condition_id'] == 0]['column_id'].values
                cond_1_columns = case_df[case_df['condition_id'] == 1]['column_id'].values
                # Require that must have cond_1 at least
                if len(cond_1_columns) > 0:
                    if len(cond_0_columns) == 0 and (label == 'CpG' or include_missing):
                        missing_cond_0_data += 1
                        cond_1_values = np.mean(df[cond_1_columns].values, axis=1)
                        vae_data_df[
                            f'{sub_condition}_{stats_cond_map[sub_condition]}_{case}_{label}-Normal'] = cond_0_mean
                        vae_data_df[
                            f'{sub_condition}_{stats_cond_map[sub_condition]}_{case}_{label}-Tumor'] = cond_1_values
                        vae_data_df[
                            f'{sub_condition}_{stats_cond_map[sub_condition]}_{case}_{label}-LogFC'] = cond_1_values - cond_0_mean
                        count_comparisons += 1
                    elif len(cond_0_columns) == 0 and label != 'CpG':
                        continue  #print("missing...", label, case)
                    else:
                        cond_0_values = np.mean(df[cond_0_columns].values, axis=1)  # May only have 1 value but this enables reps
                        cond_1_values = np.mean(df[cond_1_columns].values, axis=1)
                        vae_data_df[f'{sub_condition}_{stats_cond_map[sub_condition]}_{case}_{label}-Normal'] = cond_0_values
                        vae_data_df[f'{sub_condition}_{stats_cond_map[sub_condition]}_{case}_{label}-Tumor'] = cond_1_values
                        vae_data_df[f'{sub_condition}_{stats_cond_map[sub_condition]}_{case}_{label}-LogFC'] = cond_1_values - cond_0_values
                        count_comparisons += 1
                    # Add a few new rows to the sample df
                else:
                    missing_cond_1_data += 1
        # Set index
        vae_data_df.set_index('id', inplace=True)
        return vae_data_df