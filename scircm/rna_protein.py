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


class SciRP(SciRCM):

    def __init__(self, rna_file: str, proteomics_file: str, rna_logfc: str, rna_padj: str, prot_logfc: str,
                 prot_padj: str, gene_id: str, rna_padj_cutoff=0.05,
                 prot_padj_cutoff=0.05, rna_logfc_cutoff=1.0, prot_logfc_cutoff=0.5, output_dir='.',
                 output_filename=None, non_coding_genes=None, debug_on=False, sep=',', bg_type='P&R', sciutil=None,
                 logfile=None, reg_grp_1_lbl='RG1_All', reg_grp_2_lbl='RG2_Changes', reg_grp_3_lbl='RG3_Translation',
                 main_reg_label='RG2_Changes', reg_grp_4_lbl='RG4_Detection', check_inputs=False):
        super().__init__(None, rna_file, proteomics_file, rna_logfc, rna_padj, None, None, prot_logfc,
                         prot_padj, gene_id, rna_padj_cutoff, prot_padj_cutoff, rna_logfc_cutoff, prot_logfc_cutoff,
                         output_dir, output_filename, non_coding_genes, debug_on, sep, bg_type, sciutil, logfile,
                         reg_grp_1_lbl, reg_grp_2_lbl, reg_grp_3_lbl, main_reg_label, reg_grp_4_lbl,
                         check_inputs=check_inputs)
        self.u = SciUtil() if sciutil is None else sciutil
        plt.rcParams['svg.fonttype'] = 'none'
        self.merged_df = None
        self.prot_logfc = prot_logfc
        self.rna_logfc = rna_logfc
        self.rna_padj = rna_padj
        self.prot_padj = prot_padj
        self.gene_id = gene_id
        self.logfile = open(logfile, "w+") if logfile is not None else None  # File for logging results
        self.rna_padj_cutoff, self.prot_padj_cutoff = rna_padj_cutoff, prot_padj_cutoff
        self.rna_logfc_cutoff, self.prot_logfc_cutoff = rna_logfc_cutoff, prot_logfc_cutoff
        self.debug = debug_on
        self.output_dir = output_dir
        self.reg_grp_1_lbl = reg_grp_1_lbl
        self.reg_grp_2_lbl = reg_grp_2_lbl
        self.reg_grp_3_lbl = reg_grp_3_lbl
        self.reg_grp_4_lbl = reg_grp_4_lbl
        self.main_reg_label = main_reg_label
        if main_reg_label != reg_grp_2_lbl and main_reg_label != reg_grp_1_lbl and main_reg_label != reg_grp_3_lbl:
            self.u.err_p([f'ERROR: your main regulatory label (main_reg_label) must be one of: '
                          f'{reg_grp_1_lbl}, {reg_grp_2_lbl} or {reg_grp_3_lbl}, you passed: ', main_reg_label])
        # Otherwise this will be in the specific ones.
        if prot_logfc and rna_logfc:
            self.output_filename = output_filename if output_filename else f'scircm_r{rna_logfc_cutoff}-{rna_padj_cutoff}' \
                                                                       f'_p{prot_logfc_cutoff}-{prot_padj_cutoff}'
            if isinstance(rna_file, str):
                self.rna_df = pd.read_csv(rna_file, sep=sep)
                self.prot_df = pd.read_csv(proteomics_file, sep=sep)
            else:
                self.rna_df = rna_file
                self.prot_df = proteomics_file
        self.bg_list = ['P&R', 'P', '*', 'R', 'P|R']
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
        if len(self.prot_df[self.prot_df[self.gene_id].duplicated()]) > 0:
            num_dups = len(self.prot_df[self.prot_df[self.gene_id].duplicated()])
            self.u.warn_p(['Protein dataset contained duplicates based on ID! Dropping duplicate IDs,'
                           ' note you should do this '
                           'before running SiRCle. We have just dropped it and kept the first entry. You had: ',
                           num_dups, 'duplicates.'])
            self.prot_df = self.prot_df[~self.prot_df[self.gene_id].duplicated(keep='first')]
        if len(self.rna_df[self.rna_df[self.gene_id].duplicated()]) > 0:
            num_dups = len(self.rna_df[self.rna_df[self.gene_id].duplicated()])
            self.u.warn_p(['RNAseq dataset contained duplicates based on ID! Dropping duplicate IDs,'
                           ' note you should do this '
                           'before running SiRCle. We have just dropped it and kept the first entry. You had: ',
                           num_dups, 'duplicates.'])
            self.rna_df = self.rna_df[~self.rna_df[self.gene_id].duplicated(keep='first')]
        # Merge the dataframes together
        self.merge_dfs()

        # Calculate groups
        self.run_rcm()

        # Save the DF and return the groupings
        return self.df

    def merge_dfs(self):
        """
        Merge the supplied RNAseq, and proteomics dataframes together. We do an outer join. This can
        result in gene identifiers becoming disjoint (i.e. if there are two ensembl ID columns, we get two now).
        """
        self.rna_df = self.rna_df.set_index(self.gene_id)
        self.df = self.rna_df.copy()
        self.df = self.df.merge(self.prot_df, on=self.gene_id, how='outer', suffixes=['', '_p'])
        # Fill the rest of the values with 0's
        self.merged_df = self.df.copy()

    def run_rcm(self, rna_background=1.0, protein_background=1.0):
        protein_padjs = self.df[self.prot_padj].values
        protein_logfcs = self.df[self.prot_logfc].values
        rna_padjs = self.df[self.rna_padj].values
        rna_logfcs = self.df[self.rna_logfc].values
        reg_label_1, reg_label_2, reg_label_4, reg_label_3 = [], [], [], []
        background_filter = []
        for i, gene in enumerate(self.df[self.gene_id].values):
            rna_logfc = rna_logfcs[i]
            rna_padj = rna_padjs[i]
            prot_logfc = protein_logfcs[i]
            prot_padj = protein_padjs[i]

            rna_state, protein_state = None, None
            rna_bg, protein_bg = None, None

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

            if ~np.isnan(prot_logfc) and prot_logfc > self.prot_logfc_cutoff and prot_padj < self.prot_padj_cutoff:
                protein_state = 'Protein UP'
                protein_bg = 'threshold'
            elif ~np.isnan(prot_logfc) and prot_logfc < (- 1 * self.prot_logfc_cutoff) and prot_padj < self.prot_padj_cutoff:
                protein_state = 'Protein DOWN'
                protein_bg = 'threshold'
            elif ~np.isnan(prot_logfc) and prot_logfc < 0 and prot_padj < self.prot_padj_cutoff:  # i.e. sig down but no threshold
                protein_state = 'Protein significant-negative'
                protein_bg = 'threshold'
            elif ~np.isnan(prot_logfc) and prot_logfc > 0 and prot_padj < self.prot_padj_cutoff:  # i.e. sig up but no threshold
                protein_state = 'Protein significant-positive'
                protein_bg = 'threshold'
            # Add in a not-detected state
            elif np.isnan(prot_logfc):
                protein_state = 'Protein Undetected'
                protein_bg = 'NS'
            else:
                protein_state = 'Protein not-significant'
                if prot_padj < protein_background:
                    protein_bg = 'threshold'
                else:
                    protein_bg = 'NS'

            background = f'{rna_bg} + {protein_bg}'
            background_filter.append(background)
            state_label = f'{rna_state} + {protein_state}'

            # Same for no-change
            if state_label == 'RNA DOWN + Protein DOWN':  # State 25
                reg_label_2.append('TPDS')
                reg_label_3.append('TPDS')
                reg_label_4.append('TPDS')
            elif state_label == 'RNA No change + Protein DOWN':  # State 26
                reg_label_2.append('TMDS')
                reg_label_3.append('TMDS')
                reg_label_4.append('TMDS')
            elif state_label == 'RNA UP + Protein DOWN':  # State 27
                reg_label_2.append('TPDE_TMDS')
                reg_label_3.append('TMDS')
                reg_label_4.append('TPDE_TMDS')

            elif state_label == 'RNA DOWN + Protein Undetected':  # State 31
                reg_label_2.append('TPDS_TMDE')
                reg_label_3.append('TPDS')
                reg_label_4.append('TPDS')
            elif state_label == 'RNA No change + Protein Undetected':  # State 32
                reg_label_2.append('None')  # This would only be included if we have the non-coding groups
                reg_label_3.append('None')
                reg_label_4.append('None')
            elif state_label == 'RNA UP + Protein Undetected':  # State 33
                reg_label_2.append('TPDE_TMDS')
                reg_label_3.append('TPDE')
                reg_label_4.append('TPDE')

            elif state_label == 'RNA DOWN + Protein not-significant':  # State 28
                reg_label_2.append('TPDS_TMDE')
                reg_label_3.append('None')
                reg_label_4.append('TPDS_TMDE')
            elif state_label == 'RNA No change + Protein not-significant':  # State 29
                reg_label_2.append('None')  # This would only be included if we have the non-coding groups
                reg_label_3.append('None')
                reg_label_4.append('None')
            elif state_label == 'RNA UP + Protein not-significant':  # State 30
                reg_label_2.append('TPDE_TMDS')
                reg_label_3.append('None')
                reg_label_4.append('TPDE_TMDS')

            elif state_label == 'RNA DOWN + Protein significant-negative':  # State 25
                reg_label_2.append('TPDS_TMDE')
                reg_label_3.append('TPDS')
                reg_label_4.append('TPDS')
            elif state_label == 'RNA No change + Protein significant-negative':  # State 26
                reg_label_2.append('None')
                reg_label_3.append('None')
                reg_label_4.append('None')
            elif state_label == 'RNA UP + Protein significant-negative':  # State 27
                reg_label_2.append('TPDE_TMDS')
                reg_label_3.append('TMDS')
                reg_label_4.append('TPDE_TMDS')

            elif state_label == 'RNA DOWN + Protein significant-positive':  # State 34
                reg_label_2.append('TPDS_TMDE')
                reg_label_3.append('TMDE')
                reg_label_4.append('TPDS_TMDE')
            elif state_label == 'RNA No change + Protein significant-positive':  # State 35
                reg_label_2.append('None')
                reg_label_3.append('None')
                reg_label_4.append('None')
            elif state_label == 'RNA UP + Protein significant-positive':  # State 36
                reg_label_2.append('TPDE_TMDS')
                reg_label_3.append('TPDE')
                reg_label_4.append('TPDE')

            elif state_label == 'RNA DOWN + Protein UP':  # State 34
                reg_label_2.append('TPDS_TMDE')
                reg_label_3.append('TMDE')
                reg_label_4.append('TPDS_TMDE')
            elif state_label == 'RNA No change + Protein UP':  # State 35
                reg_label_2.append('TMDE')
                reg_label_3.append('TMDE')
                reg_label_4.append('TMDE')
            elif state_label == 'RNA UP + Protein UP':  # State 36
                reg_label_2.append('TPDE')
                reg_label_3.append('TPDE')
                reg_label_4.append('TPDE')

            else:
                reg_label_2.append('None')
                reg_label_3.append('None')
                reg_label_4.append('None')
            # Add the reg group 1
            reg_label_1.append(state_label)

        # Now we want to make three new columns that define the states of methylation RNA and protein
        self.df['RNA'] = [s if s == 'None' else s.split(' + ')[0] for s in reg_label_1]
        self.df['Protein'] = [s if s == 'None' else s.split(' + ')[1] for s in reg_label_1]

        self.df[self.reg_grp_1_lbl] = reg_label_1
        self.df[self.reg_grp_2_lbl] = reg_label_2
        self.df[self.reg_grp_3_lbl] = reg_label_3
        self.df[self.reg_grp_4_lbl] = reg_label_4

        self.df['Background_filter'] = background_filter
        # For each of the regulatory labels, apply the background filter depending on what the user chose
        self.apply_bg_filter()

    def apply_bg_filter(self):
        bg_type = self.bg_type
        bg = self.df['Background_filter'].values
        if bg_type == 'P|R':  # Protein AND (DNA methylation OR RNA)
            conds = ['threshold + threshold',
                     'threshold + NS',
                     'NS + threshold'
                     ]
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]
            self.df[f'{self.reg_grp_4_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_4_lbl].values)]

        elif bg_type == 'P&R':  # Protein and RNA
            conds = ['threshold + threshold']
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]
            self.df[f'{self.reg_grp_4_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_4_lbl].values)]

        elif bg_type == 'R':  # RNA
            conds = ['threshold + NS',
                     'threshold + threshold']
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]
            self.df[f'{self.reg_grp_4_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_4_lbl].values)]

        elif bg_type == 'P':  # Protein
            conds = ['NS + threshold',
                     'threshold + threshold'
                     ]
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]
            self.df[f'{self.reg_grp_4_lbl}_filtered'] = [c if bg[i] in conds else 'Not-Background' for i, c in enumerate(self.df[self.reg_grp_4_lbl].values)]

        elif bg_type == '*':  # Use all genes as the background
            self.df[f'{self.reg_grp_1_lbl}_filtered'] = [c for i, c in enumerate(self.df[self.reg_grp_1_lbl].values)]
            self.df[f'{self.reg_grp_2_lbl}_filtered'] = [c for i, c in enumerate(self.df[self.reg_grp_2_lbl].values)]
            self.df[f'{self.reg_grp_3_lbl}_filtered'] = [c for i, c in enumerate(self.df[self.reg_grp_3_lbl].values)]
            self.df[f'{self.reg_grp_4_lbl}_filtered'] = [c for i, c in enumerate(self.df[self.reg_grp_4_lbl].values)]
