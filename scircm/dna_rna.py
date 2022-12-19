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

import numpy as np
import pandas as pd
from sciutil import SciUtil, SciException
import matplotlib.pyplot as plt
from scircm import SciRCM


class SciRCMException(SciException):
    def __init__(self, message=''):
        Exception.__init__(self, message)


class SciMR(SciRCM):

    def __init__(self, meth_file: str, rna_file: str, rna_logfc: str, rna_padj: str, meth_diff: str, meth_padj: str,
                 gene_id: str, rna_padj_cutoff=0.05,
                 meth_padj_cutoff=0.05, rna_logfc_cutoff=1.0, meth_diff_cutoff=10, output_dir='.', output_filename=None,
                 non_coding_genes=None, debug_on=False, sep=',', bg_type='M&R', sciutil=None, logfile=None,
                 reg_grp_1_lbl='RG1_All', reg_grp_2_lbl='RG2_Changes', reg_grp_3_lbl='RG3_TwoLevels',
                 main_reg_label='RG2_Changes', check_inputs=True):
        super().__init__(meth_file, rna_file, None, rna_logfc, rna_padj, meth_diff, meth_padj, None,
                         None, gene_id, rna_padj_cutoff, meth_padj_cutoff, rna_logfc_cutoff, meth_diff_cutoff,
                         output_dir, output_filename, non_coding_genes, debug_on, sep, bg_type, sciutil, logfile,
                         reg_grp_1_lbl, reg_grp_2_lbl, reg_grp_3_lbl, main_reg_label, check_inputs=False)
        self.u = SciUtil() if sciutil is None else sciutil
        plt.rcParams['svg.fonttype'] = 'none'
        self.meth_diff = meth_diff
        self.merged_df = None
        self.rna_logfc = rna_logfc
        self.rna_padj = rna_padj
        self.meth_padj = meth_padj
        self.gene_id = gene_id
        self.logfile = open(logfile, "w+") if logfile is not None else None  # File for logging results
        self.rna_padj_cutoff, self.meth_padj_cutoff = rna_padj_cutoff, meth_padj_cutoff
        self.rna_logfc_cutoff, self.meth_diff_cutoff = rna_logfc_cutoff, meth_diff_cutoff
        self.debug = debug_on
        self.output_dir = output_dir
        self.reg_grp_1_lbl = reg_grp_1_lbl
        self.reg_grp_2_lbl = reg_grp_2_lbl
        self.main_reg_label = main_reg_label
        self.reg_grp_3_lbl = reg_grp_3_lbl
        if check_inputs:
            if main_reg_label != reg_grp_2_lbl and main_reg_label != reg_grp_1_lbl:
                self.u.err_p([f'ERROR: your main regulatory label (main_reg_label) must be one of: '
                              f'{reg_grp_1_lbl}, {reg_grp_2_lbl} or you passed: ', main_reg_label])
            # Otherwise this will be in the specific ones.
            if rna_logfc and meth_diff:
                self.output_filename = output_filename if output_filename else f'scircm_r{rna_logfc_cutoff}-{rna_padj_cutoff}' \
                                                                           f'_m{meth_diff_cutoff}-{meth_padj_cutoff}.csv'
                if isinstance(meth_file, str):
                    self.meth_df = pd.read_csv(meth_file, sep=sep)
                    self.rna_df = pd.read_csv(rna_file, sep=sep)
                else:
                    self.meth_df = meth_file
                    self.rna_df = rna_file
            self.bg_list = ['M&R', 'M|R', '*', 'M', 'R']
            if bg_type not in self.bg_list:
                self.u.err_p(['ERROR: selected background type was not allowed, please choose from one of: ', self.bg_list,
                              '\n Note: | means OR and & means AND'])
            else:
                self.bg_type = bg_type

        self.non_coding_genes = non_coding_genes
        # Contains genes for the non-coding region (use for human only).
        self.df = None
        # Contains the vae data
        self.vae = None

    def run(self):
        # First check for duplicates in IDs and drop if there are any
        if len(self.rna_df[self.rna_df[self.gene_id].duplicated()]) > 0:
            num_dups = len(self.rna_df[self.rna_df[self.gene_id].duplicated()])
            self.u.warn_p(['RNAseq dataset contained duplicates based on ID! Dropping duplicate IDs,'
                           ' note you should do this '
                           'before running SiRCle. We have just dropped it and kept the first entry. You had: ',
                           num_dups, 'duplicates.'])
            self.rna_df = self.rna_df[~self.rna_df[self.gene_id].duplicated(keep='first')]
        if len(self.meth_df[self.meth_df[self.gene_id].duplicated()]) > 0:
            num_dups = len(self.meth_df[self.meth_df[self.gene_id].duplicated()])
            self.u.warn_p(['Methylation dataset contained duplicates based on ID! Dropping duplicate IDs,'
                           ' note you should do this '
                           'before running SiRCle. We have just dropped it and kept the first entry. You had: ',
                           num_dups, 'duplicates.'])
            self.meth_df = self.meth_df[~self.meth_df[self.gene_id].duplicated(keep='first')]

        # Merge the dataframes together
        self.merge_dfs()

        # Calculate groups
        self.run_rcm()

        # Save the DF and return the groupings
        return self.df

    def merge_dfs(self):
        """
        Merge the supplied RNAseq, DNA methylation and proteomics dataframes together. We do an outer join. This can
        result in gene identifiers becoming disjoint (i.e. if there are two ensembl ID columns, we get two now).
        """
        self.rna_df = self.rna_df.set_index(self.gene_id)
        self.df = self.rna_df.merge(self.meth_df, on=self.gene_id, how='outer', suffixes=['_r', '_m'])
        # Fill the rest of the values with 0's
        self.merged_df = self.df.copy()

    def run_rcm(self, methylation_background=1.0, rna_background=1.0):
        rna_padjs = self.df[self.rna_padj].values
        rna_logfcs = self.df[self.rna_logfc].values
        meth_padjs = self.df[self.meth_padj].values
        meth_diffs = self.df[self.meth_diff].values
        reg_label_1, reg_label_2, reg_label_4, reg_label_3 = [], [], [], []
        background_filter = []
        for i, gene in enumerate(self.df[self.gene_id].values):
            meth_diff = meth_diffs[i]
            meth_padj = meth_padjs[i]
            rna_logfc = rna_logfcs[i]
            rna_padj = rna_padjs[i]
            methylation_state, rna_state = None, None
            methylation_bg, rna_bg = None, None
            if ~np.isnan(meth_diff) and meth_diff > self.meth_diff_cutoff and meth_padj < self.meth_padj_cutoff:
                methylation_state = 'Hypermethylation'
                methylation_bg = 'threshold'
            elif ~np.isnan(meth_diff) and meth_diff < (-1 * self.meth_diff_cutoff) and meth_padj < self.meth_padj_cutoff:
                methylation_state = 'Hypomethylation'
                methylation_bg = 'threshold'
            else:
                methylation_state = 'Methylation No change'
                if np.isnan(meth_padj):
                    methylation_bg = 'NS'
                elif meth_padj < methylation_background:
                    methylation_bg = 'threshold'
                else:
                    methylation_bg = 'NS'

            if ~np.isnan(rna_logfc) and rna_logfc > self.rna_logfc_cutoff and rna_padj < self.rna_padj_cutoff:
                rna_state = 'RNA UP'
                rna_bg = 'threshold'
            elif ~np.isnan(rna_logfc) and rna_logfc < (- 1 * self.rna_logfc_cutoff) and rna_padj < self.rna_padj_cutoff:
                rna_state = 'RNA DOWN'
                rna_bg = 'threshold'
            else:
                rna_state = 'RNA No change'
                if np.isnan(rna_padj):
                    rna_bg = 'NS'
                elif rna_padj < rna_background:
                    rna_bg = 'threshold'
                else:
                    rna_bg = 'NS'

            background = f'{methylation_bg} + {rna_bg}'
            background_filter.append(background)
            state_label = f'{methylation_state} + {rna_state}'

            if state_label == 'Hypermethylation + RNA DOWN':  # State 7
                reg_label_2.append('MDS')
                reg_label_3.append('MDS')
            elif state_label == 'Hypermethylation + RNA No change':  # State 8
                reg_label_2.append('None')  # This would only be included if we have the non-coding groups
                reg_label_3.append('None')
            elif state_label == 'Hypermethylation + RNA UP':  # State 9
                reg_label_2.append('TPDE')
                reg_label_3.append('MDS_TPDE')

            elif state_label == 'Hypomethylation + RNA DOWN':  # State 22
                reg_label_2.append('TPDS')
                reg_label_3.append('MDE_TPDS')
            elif state_label == 'Hypomethylation + RNA No change':  # State 23
                reg_label_2.append('None')  # This would only be included if we have the non-coding groups
                reg_label_3.append('None')
            elif state_label == 'Hypomethylation + RNA UP':  # State 24
                reg_label_2.append('MDE')
                reg_label_3.append('MDE')

            elif state_label == 'Methylation No change + RNA DOWN':  # State 31
                reg_label_2.append('TPDS')
                reg_label_3.append('TPDS')
            elif state_label == 'Methylation No change + RNA No change':  # State 32
                reg_label_2.append('None')  # This would only be included if we have the non-coding groups
                reg_label_3.append('None')
            elif state_label == 'Methylation No change + RNA UP':  # State 33
                reg_label_2.append('TPDE')
                reg_label_3.append('TPDE')
            else:
                reg_label_2.append('None')
                reg_label_3.append('None')
            # Add the reg group 1
            reg_label_1.append(state_label)

        # Now we want to make three new columns that define the states of methylation RNA and protein
        self.df['Methylation'] = [s if s == 'None' else s.split(' + ')[0] for s in reg_label_1]
        self.df['RNA'] = [s if s == 'None' else s.split(' + ')[1] for s in reg_label_1]

        self.df[self.reg_grp_1_lbl] = reg_label_1
        self.df[self.reg_grp_2_lbl] = reg_label_2
        self.df[self.reg_grp_3_lbl] = reg_label_3

        self.df['Background_filter'] = background_filter
        # For each of the regulatory labels, apply the background filter depending on what the user chose
        self.apply_bg_filter()

    def apply_bg_filter(self):
        bg_type = self.bg_type
        bg = self.df['Background_filter'].values
        if bg_type == 'M':  # Protein AND (DNA methylation OR RNA)
            conds = ['threshold + threshold',  # RNA & Methylation
                     'threshold + NS',
                     ]
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]

        elif bg_type == 'R':  # Protein OR methylation OR RNA
            conds = ['threshold + threshold',  # RNA & Methylation
                     'NS + threshold']
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]

        elif bg_type == 'M|R':  # Protein OR methylation OR RNA
            conds = ['threshold + threshold',  # RNA & Methylation
                     'NS + threshold',
                     'threshold + NS']
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]

        elif bg_type == 'M&R':  # Protein OR methylation OR RNA
            conds = ['threshold + threshold']
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]

        elif bg_type == '*':  # Use all genes as the background
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]
