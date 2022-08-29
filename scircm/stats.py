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
import random

from statsmodels.stats.multitest import multipletests
from scivae import VAE
from scipy import stats
import numpy as np
import pandas as pd
from sciutil import SciUtil
import json


class RCMStats:
    """
    Enables running a VAE on the RCM data.

    Takes in the output of rcm and then for each SiRCLe cluster, trains a VAE. It returns the dataset which can be
    used as input to calculating statistics on the DS.
    """

    def __init__(self, rcm_file: str,
                 patient_sample_file : str,
                 meth_file: str,
                 meth_sample_file: str,
                 rna_file: str,
                 rna_sample_file: str,
                 protein_file: str,
                 protein_sample_file: str,
                 output_folder: str,
                 condition_column: str,
                 column_id: str,
                 patient_id_column: str,
                 config_file: str = None,
                 regulatory_label='Regulation_Grouping_2',
                 run_name: str = None,
                 normalise='rows', verbose=False, missing_method='mean',
                 iid=False):
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
        self.patient_id_column = patient_id_column  # This needs to be the same in all cases!
        self.patient_clinical_df = pd.read_csv(patient_sample_file)
        self.meth_sample_df = pd.read_csv(meth_sample_file)
        self.rna_sample_df = pd.read_csv(rna_sample_file)
        self.protein_sample_df = pd.read_csv(protein_sample_file)
        self.condition_column = condition_column or 'condition_id'
        self.column_id = column_id or 'column_id'
        self.missing_method = missing_method # either 'mean' or 'clinical'
        # For each of these files, normalise the rows to be between 0 and 1.
        self.output_folder = output_folder
        self.encoded_df = {}  # The encoded patient data.
        self.trained_vae = {}
        self.vae_input_df = {}
        self.train_df = {}
        self.raw_input_df = {} # The raw input data (i.e. no normaliseation).
        self.normalise = normalise
        self.u = SciUtil(debug_on=verbose)  # Set this optionally for verbosity
        self.u.warn_p(["WARNING: you cannot have underscores in your case_id names. Please check this before using"
                       "this tool."])
        self.check_files()
        self.iid = iid

    def save(self):
        """
        Save relevant files.

        Returns
        -------

        """
        encoded_df = pd.DataFrame()
        input_df = pd.DataFrame()
        raw_df = pd.DataFrame()
        for reg_label in self.encoded_df:
            df = self.encoded_df[reg_label]
            df[self.regulatory_label] = reg_label
            encoded_df = encoded_df.append(df)
            df = self.vae_input_df[reg_label]
            df[self.regulatory_label] = reg_label
            input_df = input_df.append(df)
            df = self.raw_input_df[reg_label]
            df[self.regulatory_label] = reg_label
            raw_df = raw_df.append(df)
        encoded_df.to_csv(f'{self.output_folder}encoded_df_{self.run_name}.csv')
        input_df.to_csv(f'{self.output_folder}vae_input_df_{self.run_name}.csv')
        raw_df.to_csv(f'{self.output_folder}raw_input_df_{self.run_name}.csv')

        # Also save the patient info
        self.patient_clinical_df.to_csv(f'{self.output_folder}patient_info_{self.run_name}.csv', index=False)

    def load_saved_vaes(self):
        for reg_label in self.rcm[self.regulatory_label].unique():
            weight_file_path = f'{self.output_folder}{reg_label}-{self.run_name}-VAE-weights.h5'
            optimizer_file_path = f'{self.output_folder}{reg_label}-{self.run_name}-VAE-optimizer.json'
            config_json = f'{self.output_folder}{reg_label}-{self.run_name}-VAE-config.json'
            with open(config_json, "r") as fp:
                config = json.load(fp)
            vae_m = VAE(np.ones((20, len(self.feature_columns))), np.ones((20, len(self.feature_columns))),
                        list(np.ones(20)), config, vae_label=reg_label)
            # Then decode the data
            vae_m.load(weight_file_path, optimizer_file_path, config_json)
            self.trained_vae[reg_label] = vae_m

    def load_saved_inputs(self, filename):
        """
        Optionally load a saved version of the input/training data patient data. Expected to have "id" as the
        first column, then the cases and values, and the regulatory label for that gene.
        Parameters
        ----------
        filename

        Returns
        -------

        """
        input_df = pd.read_csv(filename, index_col=0)
        for reg_label in self.rcm[self.regulatory_label].unique():
            self.vae_input_df[reg_label] = input_df[input_df[self.regulatory_label] == reg_label]

    def load_saved_raws(self, filename):
        """
        Optionally load a saved version of the input/training data patient data. Expected to have "id" as the
        first column, then the cases and values, and the regulatory label for that gene.
        Parameters
        ----------
        filename

        Returns
        -------

        """
        input_df = pd.read_csv(filename, index_col=0)
        for reg_label in self.rcm[self.regulatory_label].unique():
            self.raw_input_df[reg_label] = input_df[input_df[self.regulatory_label] == reg_label]

    def load_saved_encodings(self, filename):
        """
        Optionally load a saved version of the encoded patient data. Expected to have "id" as the
        first column, then the cases and values, and the regulatory label for that gene.
        Parameters
        ----------
        filename

        Returns
        -------

        """
        input_df = pd.read_csv(filename, index_col=0)
        for reg_label in self.rcm[self.regulatory_label].unique():
            self.encoded_df[reg_label] = input_df[input_df[self.regulatory_label] == reg_label]

    def check_files(self):
        """
        We need to check the loaded files to make sure they are in the correct format.

        Checks:
            1. The index of each of the data files overlaps with the regulatory clusters.
            2. Each sample file must have a) condition_column (default = 'condition_id') which has values 1 = tumour or
            0 = Normal and b) a column ID field (default = 'column_id') that maps to the column of that dataset,
            c) a case ID (that is a patient ID: default='case_id' that maps to a patient ID in the patient_clinical_df
            which has clinical information about the patient. case_ids can't have underscores in them.
        Returns
        -------

        """
        # Check 1.
        rcm_genes = set(self.rcm.index.values)
        protein_genes = set(self.protein_data.index.values)
        rna_genes = set(self.rna_data.index.values)
        meth_genes = set(self.meth_data.index.values)
        total_overlap = len(rcm_genes & rna_genes & protein_genes & meth_genes)
        self.u.dp(['Overlap between SiRCLe gene clusters and protein, RNA, and CpG: ', total_overlap, '\n',
                   f'If you used SiRCLe, this should be the total number of genes in your SiRCLe dataset. '
                   f'{len(rcm_genes)}', '\n First 5 gene IDs in RCM: ', list(rcm_genes)[0:5],
                   '\n First 5 gene IDs in Protein: ', list(protein_genes)[0:5],
                   '\n First 5 gene IDs in RNA: ', list(rna_genes)[0:5],
                   '\n First 5 gene IDs in Methylation: ', list(meth_genes)[0:5],
                   ])
        if total_overlap == 0:
            self.u.err_p(['You had no overlap between your gene identifiers! That is not good, nothing will work.'
                          'Please make sure your gene IDs are matching and are in the first column of your CSV file.'
                          '\nAlso ensure you are using a CSV not a TSV.'])
            return

        # Part 2.
        if not self.check_cols_exist():
            return

        # Part 3. Check for duplicates
        if len(self.protein_data[self.protein_data.index.duplicated()]) > 0:
            num_dups = len(self.protein_data[self.protein_data.index.duplicated()])
            self.u.warn_p(['Protein dataset contained duplicates! Dropping duplicate IDs, note you should do this '
                           'before running SiRCle. We have just dropped it and kept the first entry. You had: ',
                           num_dups, 'duplicates.'])
            self.protein_data = self.protein_data[~self.protein_data.index.duplicated(keep='first')]
        if len(self.rna_data[self.rna_data.index.duplicated()]):
            num_dups = len(self.rna_data[self.rna_data.index.duplicated()])
            self.u.warn_p(['RNA dataset contained duplicates! Dropping duplicate IDs, note you should do this '
                           'before running SiRCle. We have just dropped it and kept the first entry. You had: ',
                           num_dups, 'duplicates.'])
            self.rna_data = self.rna_data[~self.rna_data.index.duplicated(keep='first')]
        if len(self.meth_data[self.meth_data.index.duplicated()]) > 0:
            num_dups = len(self.meth_data[self.meth_data.index.duplicated()])
            self.u.warn_p(['DNA Methylation dataset contained duplicates! Dropping duplicate IDs,'
                           ' note you should do this before running SiRCle. We have just dropped it and '
                           'kept the first entry You had: ',
                           num_dups, 'duplicates.'])
            self.meth_data = self.meth_data[~self.meth_data.index.duplicated(keep='first')]
        self.build_sample_df()

    def get_sample_column(self, sample_df, case_id, condition_id):
        """
        Get the column identifier for a given case, sample df and condition ID.

        Parameters
        ----------
        sample_df
        case_id
        condition_id

        Returns
        -------

        """
        case_sample_df = sample_df[sample_df[self.patient_id_column] == case_id]
        if len(sample_df) > 0:
            value = case_sample_df[case_sample_df[self.condition_column] == condition_id][self.column_id].values
            if len(value) > 0:
                # only ever take one
                if len(value) > 1:
                    self.u.warn_p(['Had multiple samples for: ', case_id, 'just took the first one.',
                                   value])
                return value[0]
        return None

    def build_sample_df(self):
        """
        Builds a sample DF containing the columns that refer to the tumour and normal samples for the dataframes.
        This will help build the comparisons later on and also will make it clear which patients have had missing
        data added in.

        Returns
        -------

        """
        # Part 3. building the patient dataframe and making sure we have good data there (i.e. matching patients).
        protein_tumour = []
        rna_tumour = []
        meth_tumour = []
        protein_normal = []
        rna_normal = []
        meth_normal = []
        counts = []
        for i, case_id in enumerate(self.patient_clinical_df[self.patient_id_column].values):
            protein_normal.append(self.get_sample_column(self.protein_sample_df, case_id, 0))
            protein_tumour.append(self.get_sample_column(self.protein_sample_df, case_id, 1))
            rna_normal.append(self.get_sample_column(self.rna_sample_df, case_id, 0))
            rna_tumour.append(self.get_sample_column(self.rna_sample_df, case_id, 1))
            meth_normal.append(self.get_sample_column(self.meth_sample_df, case_id, 0))
            meth_tumour.append(self.get_sample_column(self.meth_sample_df, case_id, 1))
            # Add the number of non NAs we got
            count_data = 1 if protein_normal[i] is not None else 0
            count_data += 1 if protein_tumour[i] is not None else 0
            count_data += 1 if rna_normal[i] is not None else 0
            count_data += 1 if rna_tumour[i] is not None else 0
            count_data += 1 if meth_normal[i] is not None else 0
            count_data += 1 if meth_tumour[i] is not None else 0
            counts.append(count_data)
        df = self.patient_clinical_df.copy()
        # Make a new sample df incorperating this info.
        df['Protein Tumour'] = protein_tumour
        df['Protein Normal'] = protein_normal
        df['RNA Tumour'] = rna_tumour
        df['RNA Normal'] = rna_normal
        df['CpG Tumour'] = meth_tumour
        df['CpG Normal'] = meth_normal
        df['Sample counts'] = counts
        self.patient_clinical_df = df  # update possibly this is a bad idea...
        return df

    def check_cols_exist(self):
        """
        Check required columns exist in the sample data frames.
        Required columns:
        Each sample file must have a) condition_column (default = 'condition_id') which has values 1 = tumour or
            0 = Normal and b) a column ID field (default = 'column_id') that maps to the column of that dataset,
            c) a case ID (that is a patient ID: default='case_id' that maps to a patient ID in the patient_clinical_df
            which has clinical information about the patient. case_ids can't have underscores in them.
        Returns
        -------

        """
        cols = [self.condition_column, self.column_id, self.patient_id_column]
        # Check 2.
        for required_column in cols:
            if required_column not in self.meth_sample_df.columns:
                self.u.err_p([f'Your {required_column} was not in your methylation sample file?',
                              'This is needed. Nothing will work. Make sure your sample file is a CSV file.\n',
                              'Columns in the file you passed: ', self.meth_sample_df.columns])
                return False
            if required_column not in self.rna_sample_df.columns:
                self.u.err_p([f'Your {required_column} was not in your RNA sample file?',
                              'This is needed. Nothing will work. Make sure your sample file is a CSV file.\n',
                              'Columns in the file you passed: ', self.rna_sample_df.columns])
                return False

            if required_column not in self.protein_sample_df.columns:
                self.u.err_p([f'Your {required_column} was not in your Protein sample file?',
                              'This is needed. Nothing will work. Make sure your sample file is a CSV file.\n',
                              'Columns in the file you passed: ', self.protein_sample_df.columns])
                return False
        # Lastly check that the column id actually has overlaps with their data and they haven't done something dumb.
        protein_cols = set(self.protein_data.columns) & set(self.protein_sample_df[self.column_id].values)
        rna_cols = set(self.rna_data.columns) & set(self.rna_sample_df[self.column_id].values)
        methylation_cols = set(self.meth_data.columns) & set(self.meth_sample_df[self.column_id].values)
        if len(protein_cols) == 0:
            self.u.err_p([f'You made a mistake...  {self.column_id} does not actually map to your protein data file?',
                          'columns in your protein data file:', self.protein_data.columns,
                          '\n vs columns in your protein sample file: ', self.protein_sample_df[self.column_id].values])
            return False
        if len(rna_cols) == 0:
            self.u.err_p([f'You made a mistake...  {self.column_id} does not actually map to your protein data file?',
                          'columns in your protein data file:', self.rna_data.columns,
                          '\n vs columns in your protein sample file: ', self.rna_sample_df[self.column_id].values])
            return False

        if len(methylation_cols) == 0:
            self.u.err_p([f'You made a mistake...  {self.column_id} does not actually map to your protein data file?',
                          'columns in your protein data file:', self.meth_data.columns,
                          '\n vs columns in your protein sample file: ', self.meth_sample_df[self.column_id].values])
            return False

        # Final, check that there were overlapping patients in all and in the patient dataframe.
        patients_overlap = set(self.patient_clinical_df[self.patient_id_column].values) & \
                           set(self.protein_sample_df[self.patient_id_column].values) & \
                           set(self.rna_sample_df[self.patient_id_column].values) & \
                           set(self.meth_sample_df[self.patient_id_column].values)
        if len(patients_overlap) == 0:
            self.u.err_p([f'Hmmm your patient ids: ', self.patient_id_column, 'did not map in one of your sample files',
                          ' or the patient clinical info, please check them and then try again.'])
            return False

        # Sanitise the columns (replace '_' with .)
        self.patient_clinical_df[self.patient_id_column] = [c.replace('_', '.') for c in self.patient_clinical_df[self.patient_id_column].values]
        self.protein_sample_df[self.patient_id_column] = [c.replace('_', '.') for c in self.protein_sample_df[self.patient_id_column].values]
        self.rna_sample_df[self.patient_id_column] = [c.replace('_', '.') for c in self.rna_sample_df[self.patient_id_column].values]
        self.meth_sample_df[self.patient_id_column] = [c.replace('_', '.') for c in self.meth_sample_df[self.patient_id_column].values]

        # Yay they did good.
        return True

    def run_vae_stats(self, cond_label: str, cond0: str, cond1: str, label='', selected_cases=None, test_type='mannwhitneyu'):
        """
        Run stats comparing samples with condition 1 vs condition 0, the cond_label column. This is a column
        that must be present in all the sample data frames (for example, "gender" or "stage).

        Parameters
        ----------
        cond_label: the label of column in the patient sample df of
            the condition that we want to do that stats on e.g. gender, or stage
        cond0: the value that we want to be the control (e.g. stage 1)
        cond1: the value we want to test deviates from the control (e.g. stage 4)
        label: the label for this test
        include_missing: whether you want to include patients that are missing 1 or more data values e.g. only containes
            tumour for the protein data as opposed to both tumour and normal.
            If this is ticked, the data is filled with the mean value for that condition.

        Returns
        -------

        """
        cond1_cases_all = self.patient_clinical_df[self.patient_clinical_df[cond_label] == cond1][self.patient_id_column].values
        cond0_cases_all = self.patient_clinical_df[self.patient_clinical_df[cond_label] == cond0][self.patient_id_column].values

        if selected_cases is not None:
            cond1_cases_all = [c for c in cond1_cases_all if c in selected_cases]
            cond0_cases_all = [c for c in cond0_cases_all if c in selected_cases]

        # Note since not all of these values may have been included, keep only the cases that also were in the input df
        all_stats = pd.DataFrame()
        for reg_label in self.rcm[self.regulatory_label].unique():
            encoded_data = self.encoded_df[reg_label]  # Get the pre-encoded data for these patients...

            # Get it for each of the columns to align to
            cols_to_align = ['Protein-LogFC', 'RNA-LogFC', 'CpG-LogFC']
            cond0_cases = list(set([col for col in encoded_data.columns if col in cond0_cases_all]))
            cond1_cases = list(set([col for col in encoded_data.columns if col in cond1_cases_all]))

            alignment_column_1_values = []
            alignment_column_0_values = []
            # Also we want to make sure the columns are aligned to something biologicallly meaningful, so we want
            # to add in their "input data" so it can be aligned to this and we do this on the non-normalised data
            raw_input_df = self.raw_input_df[reg_label]
            for c in cols_to_align:
                # Get the mean value for this condition
                cols = [col for col in raw_input_df.columns if c in col and col.split('_')[0] in cond1_cases]
                data = np.nanmean(raw_input_df[cols].values, axis=1)
                alignment_column_1_values.append(data)
                cols = [col for col in raw_input_df.columns if c in col and col.split('_')[0] in cond0_cases]
                data = np.nanmean(raw_input_df[cols].values, axis=1)
                alignment_column_0_values.append(data)

            # cond_0_encodings = {case_id: [encoded_data[case_id].values] for case_id in cond0_cases}
            # cond_1_encodings = {case_id: [encoded_data[case_id].values] for case_id in cond1_cases}
            stats_df = self.make_stats_df(test_type=test_type, id_vals=encoded_data['id'].values,
                                          cond_1_encodings=encoded_data[cond1_cases],
                                          cond_0_encodings=encoded_data[cond0_cases],
                                          column_to_align_to=cols_to_align,
                                          alignment_column_1_values=alignment_column_1_values,
                                          alignment_column_0_values=alignment_column_0_values, cond0=cond0, cond1=cond1)
            stats_df[self.regulatory_label] = reg_label
            # Save the averages from the cols to align to as well
            for i, c in enumerate(cols_to_align):
                stats_df[f'{c} mean ({cond1})'] = alignment_column_1_values[i]
                stats_df[f'{c} mean ({cond0})'] = alignment_column_0_values[i]
                stats_df[f'{c} mean ({cond1}-{cond0})'] = alignment_column_1_values[i] - alignment_column_0_values[i]

            self.test_for_normality(stats_df[f'Integrated mean ({cond0})'], f'{reg_label} Integrated mean ({cond0})')
            self.test_for_normality(stats_df[f'Integrated mean ({cond1})'], f'{reg_label} Integrated mean ({cond1})')
            all_stats = all_stats.append(stats_df)
        all_stats.to_csv(f'{self.output_folder}stats_{cond1}-{cond0}_{self.run_name + label}.csv')
        return all_stats

    def test_for_normality(self, values, label, test_type: str = "shapiro"):
        """ Perform a test for normality."""
        k2, p = stats.normaltest(values)
        if p < 0.05:  # null hypothesis: x comes from a normal distribution
            print(f'{label}: NOT normally distributed')
            return False
        return True

    def make_stats_df(self, test_type, id_vals, cond_1_encodings, cond_0_encodings, column_to_align_to,
                      alignment_column_1_values, alignment_column_0_values, cond0, cond1):
        # Now we want to perform the differential test on the data between cond 1 - cond 0
        # If we have multiple samples we need to do this for each one
        if len(id_vals) > 0:
            stat_vals = []
            p_vals = []
            base_means_cond_0 = []
            base_means_cond_1 = []
            num_cond_0 = 0
            num_cond_1 = 0

            # For each case in the encodings we want to collect the values
            for i in range(0, len(id_vals)):
                # ToDo: extend to anova or other statistical tests for more data types.
                cases_0_vals = cond_0_encodings.values[i]
                cases_1_vals = cond_1_encodings.values[i]
                num_cond_0 = len(cases_0_vals)
                num_cond_1 = len(cases_1_vals)
                # potentially wrap a try catch if there are all even numbers
                if test_type == 't-test':
                    t_stat, p_val = stats.ttest_ind(cases_1_vals, cases_0_vals)
                else:
                    t_stat, p_val = stats.mannwhitneyu(cases_1_vals, cases_0_vals)
                if p_val == 0 or p_val > 1:
                    p_val = 1.0
                stat_vals.append(t_stat)
                p_vals.append(p_val)
                base_mean_cond_1 = np.nanmean(cases_1_vals)
                base_mean_cond_0 = np.nanmean(cases_0_vals)
                base_means_cond_0.append(base_mean_cond_0)
                base_means_cond_1.append(base_mean_cond_1)
            # Now we have the p-values we can perform the correction
            reg, corrected_p_vals, a, b = multipletests(p_vals, method='fdr_bh', alpha=0.05, returnsorted=False)
            # Return something similar to what you'd get from DEseq2
            stats_df = pd.DataFrame()
            stats_df['id'] = id_vals
            stats_df[f'{test_type} stat ({cond1}-{cond0})'] = stat_vals
            stats_df[f'Integrated padj ({cond1}-{cond0})'] = corrected_p_vals
            stats_df[f'Integrated pval ({cond1}-{cond0})'] = p_vals
            # Check if we have a column to align to
            base_means_cond_1 = np.array(base_means_cond_1)
            base_means_cond_0 = np.array(base_means_cond_0)
            if column_to_align_to is not None:
                # Go through each one and stop if we get over 0.5 correlation
                for col_i in range(0, len(alignment_column_0_values)):
                    mean_col_0 = alignment_column_0_values[col_i]  # Across genes
                    mean_col_1 = alignment_column_1_values[col_i]
                    col_0_corr = np.corrcoef(mean_col_0, base_means_cond_0)[0, 1]
                    col_1_corr = np.corrcoef(mean_col_1, base_means_cond_1)[0, 1]
                    if abs(col_0_corr) > 0.5 or abs(col_1_corr) > 0.5:
                        if abs(col_0_corr) > abs(col_1_corr):
                            direction = -1 if col_0_corr < 0 else 1
                        else:
                            direction = -1 if col_1_corr < 0 else 1
                        # Convert both
                        base_means_cond_0 = direction * base_means_cond_0
                        base_means_cond_1 = direction * base_means_cond_1
                        break # If none of them meet it then we don't change anything
            # Compute difference as the distance between the two
            distances = []
            for i, cond_0 in enumerate(base_means_cond_0):
                if cond_0 < 0:
                    distances.append(base_means_cond_1[i] + abs(cond_0))
                else:
                    distances.append(base_means_cond_1[i] - abs(cond_0))
            stats_df[f'Integrated diff ({cond1}-{cond0})'] = distances
            stats_df[f'Integrated mean ({cond0})'] = base_means_cond_0
            stats_df[f'Integrated mean ({cond1})'] = base_means_cond_1
            self.u.dp(['Summary\n', f'Cond1: {num_cond_1} vs Cond0: {num_cond_0}\n',
                       stats_df.describe()])
            # Also make a copy that also contains all the info from all the cases
            # make this optional later on...
            for c in cond_0_encodings:
                stats_df[f'{cond0}_{c}'] = cond_0_encodings[c].values
            for c in cond_1_encodings:
                stats_df[f'{cond1}_{c}'] = cond_1_encodings[c].values
            return stats_df
        else:
            # Only one value so just do the test once.
            cases_0_vals = [c for c in cond_0_encodings.values]
            cases_1_vals = [c for c in cond_1_encodings.values]
            t_stat, p_val = stats.mannwhitneyu(cases_1_vals, cases_0_vals)
            return t_stat, p_val

    def merge_data(self, protein_data, rna_data, meth_data):
        return pd.concat([protein_data, rna_data, meth_data], axis=1)

    def train_vae(self, cases, config=None, include_missing=True):
        """

        Parameters
        ----------
        cases: a list of patient identifiers which should be used for training, these are ideally, high quality with
        no missing data.
        config: a dictionary of configuration for the VAE training
        include_missing: whether or not to include patients that are missing some data (it is filled in by the mean value)
        note we only allow missing in terms of the "normal" not the tumour.
        Returns
        -------

        """
        encoded_df = pd.DataFrame()
        for reg_label in self.rcm[self.regulatory_label].unique():
            if reg_label != "None":
                rcm_df = self.rcm[self.rcm[self.regulatory_label] == reg_label].copy()
                meth_data = self.compute_columns_training(self.align_to_rcm(self.meth_data, rcm_df), self.meth_sample_df, 'CpG',
                                                          include_missing)
                rna_data = self.compute_columns_training(self.align_to_rcm(self.rna_data,  rcm_df), self.rna_sample_df, 'RNA',
                                                         include_missing)
                protein_data = self.compute_columns_training(self.align_to_rcm(self.protein_data, rcm_df,
                                                                               include_regulatory_label=True),
                                                             self.protein_sample_df,
                                                             'Protein', include_missing)

                r_df = self.merge_data(protein_data, rna_data, meth_data)
                self.vae_input_df[reg_label] = r_df.copy()  # Keep track of the training dataframe for these patients
                # Also do the same for a non-normalised version, we want this so that we can actually get the change
                # in protein etc.
                nn_meth_data = self.compute_columns_training(self.align_to_rcm(self.meth_data, rcm_df, normalise=False),
                                                          self.meth_sample_df, 'CpG',
                                                          include_missing)
                nn_rna_data = self.compute_columns_training(self.align_to_rcm(self.rna_data, rcm_df, normalise=False), self.rna_sample_df,
                                                         'RNA',
                                                         include_missing)
                nn_protein_data = self.compute_columns_training(self.align_to_rcm(self.protein_data, rcm_df, normalise=False,
                                                                               include_regulatory_label=True),
                                                             self.protein_sample_df,
                                                             'Protein', include_missing)
                self.raw_input_df[reg_label] = self.merge_data(nn_protein_data, nn_rna_data, nn_meth_data)
                # Now we need to filter out patients that didn't have the required matching data.
                train_df = self.build_training_df(r_df, selected_cases=cases)
                self.train_df[reg_label] = train_df.copy()
                self.train(train_df, self.feature_columns, reg_label, config)
                # Encode data with trained VAE for all patients.
                # encoding has genes as the rows IDs and patients as the columns, so we're basically building it up
                # for each of the patients
                reg_encoded_df = self.get_encoding(r_df, reg_label)
                reg_encoded_df[self.regulatory_label] = reg_label
                # Keep track of this for quick access
                self.encoded_df[reg_label] = reg_encoded_df
                encoded_df = encoded_df.append(reg_encoded_df)
        # Keep track of the patient encodings.
        return encoded_df

    def build_training_df(self, df, filter_extremes=True, selected_cases=None):
        """
        Selected cases are the cases with matched tumour and normal for all conditions (or a selection
        of cases that are used for training). This should be set otherwise all cases will be used and that
        could be suboptimal.
        Parameters
        ----------
        df: dataframe
        filter_extremes: whether or not to remove data (genexpatient) if that value is > 2 out of z score true by default
        selected_cases: the cases that are going to be used for training (list)

        Returns
        -------

        """
        # We want to add all the case data as training data and just keep the columns in the correct order
        # Basically we do this for all cases.
        cases = self.patient_clinical_df[self.patient_id_column].values
        train_df = pd.DataFrame()
        case_ids = []
        included_cases = []

        if selected_cases is not None:
            cases_subset = [c for c in cases if c in selected_cases]
            cases = cases_subset
        case_genes = None
        if self.iid:
            # Here is for the person with lots of money and resources who could actually make a big enough dataset
            # Since I think the above works, maybe next time spend your money on better things like make a new journal
            # where it's free to publish or the scientists actually get the royalties or something ya know...
            # To do this, we take a single gene from each patient, i.e. first collect the patients, then subsample
            # a single row that is unique between.
            patients = cases #list(set([c.split('_')[0] for c in df.columns]))
            n_genes = len(patients)
            if n_genes > len(df.index.values) or self.iid > len(df.index.values):
                n_genes = len(df.index.values) - 1
            # Randomly select the a sample of genes for that patient
            case_genes = []
            for case_idx, case in enumerate(cases):
                genes_for_patient = random.sample(list(enumerate(df.index.values)), n_genes)
                case_genes.append([[c[0] for c in genes_for_patient], [c[1] for c in genes_for_patient]])
            #case_genes = random.sample(list(enumerate(df.index.values)), n_genes)
        for case_idx, case in enumerate(cases):
            try:
                case_cond_df = pd.DataFrame()
                if self.iid:
                    idval = case_genes[case_idx][1]
                    case_cond_df['id'] = idval
                else:
                    case_cond_df['id'] = list(df.index.values)
                for col in self.feature_columns:

                        if self.iid:
                            valval = case_genes[case_idx][0]
                            v = df[f'{case}_{col}'].values[valval]
                            # Just select the single gene that we're interested in
                            case_cond_df[col] = v
                        else:
                            case_cond_df[col] = df[f'{case}_{col}'].values  # Get the column name from the case

                # Add this to the cond_1_sample_df
                if len(case_cond_df.columns) == len(self.feature_columns) + 1:  # For the index column
                    train_df = train_df.append(case_cond_df)
                    # Add the length of this to the case_ids list so we can extract this patient's information later
                    case_ids += [case] * len(case_cond_df)
                    included_cases.append(case)
            except:
                continue
        train_df.set_index('id', inplace=True)
        if filter_extremes:
            z_score = np.abs(stats.zscore(train_df[self.feature_columns].values, axis=1))
            max_z_score = np.max(z_score, axis=1)
            train_df = train_df[max_z_score < 2]
        self.u.dp([f'{len(included_cases)} had matched data.', included_cases])
        return train_df

    def normalise_df(self, df):
        """
        Normalise the dataframe either at a row level or a column level.

        Parameters
        ----------
        df

        Returns
        -------

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

    def fill_missing(self, df, case_id, sample_df, label):
        """
        If the user wants to include the misisng data, do it either by the mean, or by a series of the
        values associated with the patient sample types.

        Parameters
        ----------
        case_id

        Returns
        -------

        """
        if self.missing_method == 'mean':
            cond_0 = list(sample_df[sample_df[self.condition_column] == 0][self.column_id].values)
            cond_0_cols = [c for c in cond_0 if c in list(df.columns)]  # Ensure it is in the cols
            return np.mean(df[cond_0_cols].values, axis=1)
        if self.missing_method == 'clinical':
            # For now just do it on patient stage and age of patient
            case_info = self.patient_clinical_df[self.patient_clinical_df[self.patient_id_column] == case_id]
            # Get the information we care about ToDo: generalise
            age = case_info['AgeGrouped'].values[0]
            stage = case_info['Stage'].values[0]
            # Now let's get the values for that age and stage
            other_cases = self.patient_clinical_df[self.patient_clinical_df['AgeGrouped'] == age]
            other_cases = other_cases[other_cases['Stage'] == stage]
            # Now return the values for this for the given label that was asked for.
            cond_0_columns = [c for c in other_cases[f'{label} Normal'].values if c in df.columns]  # The normal values for the cases with same
            # age and stage
            return np.mean(df[cond_0_columns].values, axis=1)

    def compute_columns_training(self, df, sample_df, label, include_missing):
        """
        Goal is to add in the data for the columns for the VAE training. For this context we're interested
        in the tumour, normal, and logFC between the two.
        Parameters
        ----------
        df
        label
        include_missing

        Returns
        -------

        """
        vae_data_df = pd.DataFrame()  # This will be the data used for input
        vae_data_df['id'] = df.index.values
        # Get a baseline for all "normal"
        cond_0 = sample_df[sample_df[self.condition_column] == 0]  # i.e. normal
        cond_0_columns_all = [c for c in cond_0[self.column_id] if c is not None]
        if len(cond_0_columns_all) == 0:
            self.u.warn_p(['Dataset passed had no normal columns... you have an error please fix it!'])
            return
        # baseline.
        for case in self.patient_clinical_df[self.patient_id_column].unique():
            # Want to do condition 1 - condition 0 --> keeping with standard approach
            case_df = self.patient_clinical_df[self.patient_clinical_df[self.patient_id_column] == case]
            cond_0_column = case_df[f'{label} Normal'].values[0]
            cond_1_column = case_df[f'{label} Tumour'].values[0]
            # Require that must have cond_1 at least
            if cond_1_column is not None:
                if cond_0_column is None:
                    if include_missing:
                        cond_0_mean = self.fill_missing(df, case, sample_df, label)
                        # If it doesn't have any NA, include this otherwise ommit the data
                        if not np.isnan(cond_0_mean).any():
                            cond_1_values = df[cond_1_column].values
                            vae_data_df[f'{case}_{label}-Normal'] = cond_0_mean
                            vae_data_df[f'{case}_{label}-Tumor'] = cond_1_values
                            vae_data_df[f'{case}_{label}-LogFC'] = cond_1_values - cond_0_mean
                else:
                    # May only have 1 value for a patient - this summarises the replicates
                    cond_0_values = df[cond_0_column].values
                    cond_1_values = df[cond_1_column].values
                    vae_data_df[f'{case}_{label}-Normal'] = cond_0_values
                    vae_data_df[f'{case}_{label}-Tumor'] = cond_1_values
                    vae_data_df[f'{case}_{label}-LogFC'] = cond_1_values - cond_0_values
        # Set index
        vae_data_df.set_index('id', inplace=True)
        return vae_data_df

    def train(self, train_df, feature_columns, reg_label, config=None):
        """
        Train the vae.
        Parameters
        ----------
        train_df: a dataframe of rows which are patient gene values (i.e. n rows = patients x genes) and summarised
        columns (i.e. RNA logFC, Protein logFC, ... etc for that patient for that gene).
        feature_columns: the columns used for the VAE.
        reg_label: the SiRCle cluster.
        config: a dictionary of the config for the VAE. If None, then a default is used.

        Returns
        -------

        """
        if config is None:
            config = {"loss": {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 0.25},
                      "encoding": {"layers": [{"num_nodes": 5, "activation_fn": "relu"}]},
                      "decoding": {"layers": [{"num_nodes": 5, "activation_fn": "relu"}]},
                      "latent": {"num_nodes": 1},
                      "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                      "epochs": 200,
                      "batch_size": 16,
                      "scale_data": False
                      }
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 16)
        data_values = train_df[feature_columns].values
        vae_m = VAE(data_values, data_values, list(train_df.index.values), config, vae_label=reg_label)
        vae_m.encode('default', train_percent=75.0, epochs=epochs, batch_size=batch_size,
                     logging_dir=self.output_folder,
                     logfile=f'VAE-logfile-{reg_label}-{self.run_name}.txt', early_stop=True)
        vae_m.save(weight_file_path=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-weights.h5',
                   optimizer_file_path=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-optimizer.json',
                   config_json=f'{self.output_folder}{reg_label}-{self.run_name}-VAE-config.json')  # save the VAE
        # Save an encoding of the data in the class and return it to the user.
        self.trained_vae[reg_label] = vae_m
        vae_m.u.dp(["Saved VAE to current directory."])
        return vae_m

    def get_decoding(self, reg_label):
        if not self.trained_vae.get(reg_label):
            self.u.err_p(['That regulatory label:', reg_label, 'did not exist in your dataset, please check the csv'
                                                               'file to make sure it exists.',
                          'Regulatory labels in your dataset:', list(self.trained_vae.keys())])
        else:
            vae_m = self.trained_vae[reg_label]
            # Get the encoded data for that label
            encoding = self.encoded_df[reg_label]
            decoding_df = encoding[['id', self.regulatory_label]].copy()
            for column in encoding.columns:
                if column != 'id' and column != self.regulatory_label:
                    encoding_vals = encoding[column].values
                    decoding = vae_m.decoder.predict(np.array([np.array(c) for c in encoding_vals]))
                    # The decoding will be in the same format as what we in
                    for i, c in enumerate(self.feature_columns):
                        decoding_df[f'{column}_{c}'] = decoding[:, i]
            return decoding_df

    def get_encoding(self, input_df, reg_label):
        # Do this for each case, and then save as a DF for this cluster
        r_df = pd.DataFrame()
        r_df['id'] = input_df.index
        # Just make sure everything is in the right order
        vae_m = self.trained_vae[reg_label]
        for case_id in self.patient_clinical_df[self.patient_id_column].unique():
            case_cond_df = pd.DataFrame()
            for col in self.feature_columns:
                try:
                    case_cond_df[col] = input_df[f'{case_id}_{col}'].values  # Get the column name from the case
                except:
                    continue
            if len(case_cond_df.columns) == len(self.feature_columns):
                encoded_data = vae_m.encode_new_data(case_cond_df.values, scale=False)
                # For each of the cases add it as a column
                r_df[case_id] = encoded_data
            # else:
            #     self.u.dp(["CaseID: ", case_id, " must have been missing tumour data..."])
        self.encoded_df[reg_label] = r_df  # Save to state as well while we're going.
        return r_df

    def align_to_rcm(self, df, rcm_df, include_regulatory_label=False, normalise=True):
        aligned_df = pd.DataFrame()
        aligned_df['genes'] = rcm_df.index
        aligned_df.set_index('genes', inplace=True)
        aligned_df = aligned_df.join(df, how='left')  # Get all the data aligned to the same index
        aligned_df.fillna(0, inplace=True)  # Fill in the missing values with 0
        # normalise df
        if normalise:
            normalised_aligned_df = self.normalise_df(aligned_df) # normalise the data
            if include_regulatory_label:
                normalised_aligned_df[self.regulatory_label] = rcm_df[self.regulatory_label].values
            return normalised_aligned_df
        else:
            if include_regulatory_label:
                aligned_df[self.regulatory_label] = rcm_df[self.regulatory_label].values
                return aligned_df
            return aligned_df
