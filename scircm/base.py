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
import os
from sciutil import SciUtil, SciException
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues

class SciRCMException(SciException):
    def __init__(self, message=''):
        Exception.__init__(self, message)

"""
Sci-RCM is the logical regulatory clustering of genes based on DNA-methylation, RNA-seq and Proteomics data.

| Methylation      | RNAseq    | Proteomics | Regulation driver_1          | Regulation driver_2     | Regulation_Grouping1 | Regulation_Grouping2 | Regulation_Grouping3 |
|------------------|-----------|------------|------------------------------|-------------------------|----------------------|----------------------|----------------------|
| Hypermethylation | DOWN      | DOWN       | Methylation increase (MDS)   | None                    | MDS                  | MDS                  | MDS                  |
| Hypermethylation | UP        | DOWN       | mRNA increase (TPDE)         | Protein decrease (TMDS) | TPDE+TMDS            | TPDE+TMDS            | TMDS                 |
| Hypermethylation | UP        | UP         | mRNA increase (TPDE)         | None                    | TPDE                 | TPDE                 | TPDE                 |
| Hypermethylation | DOWN      | UP         | Methylation increase (MDS)   | Protein increase (TMDE) | MDS+TMDE             | TMDE                 | TMDE                 |
| Hypermethylation | No Change | UP         | mRNA increase (TPDE)         | Protein increase (TMDE) | TPDE+TMDE            | TMDE                 | TMDE                 |
| Hypermethylation | No Change | DOWN       | mRNA increase (TPDE)         | Protein decrease (TMDS) | TPDE+TMDS            | TMDS                 | TMDS                 |
| Hypermethylation | UP        | No Change  | mRNA increase (TPDE)         | Protein decrease (TMDS) | TPDE+TMDS            | TPDE+TMDS            | TMDS                 |
| Hypermethylation | DOWN      | No Change  | Methylation increase (MDS)   | Protein increase (TMDE) | MDS+TMDE             | MDS+TMDE             | TMDE                 |
| Hypermethylation | No Change | No Change  | Methylation increase (ncRNA) | None                    | MDS-ncRNA            | MDS_ncRNA            | MDS_ncRNA            |
| Hypomethylation  | DOWN      | DOWN       | mRNA decrease (TPDS)         | None                    | TPDS                 | TPDS                 | TPDS                 |
| Hypomethylation  | UP        | DOWN       | Methylation decrease (MDE)   | Protein decrease (TMDS) | MDE+TMDS             | TMDS                 | TMDS                 |
| Hypomethylation  | UP        | UP         | Methylation decrease (MDE)   | None                    | MDE                  | MDE                  | MDE                  |
| Hypomethylation  | DOWN      | UP         | mRNA decrease (TPDS)         | Protein increase (TMDE) | TPDS+TMDE            | TPDS+TMDE            | TMDE                 |
| Hypomethylation  | No Change | UP         | mRNA decrease (TPDS)         | Protein increase (TMDE) | TPDS+TMDE            | TMDE                 | TMDE                 |
| Hypomethylation  | No Change | DOWN       | mRNA decrease (TPDS)         | Protein decrease (TMDS) | TPDS+TMDS            | TMDS                 | TMDS                 |
| Hypomethylation  | UP        | No Change  | Methylation decrease (MDE)   | Protein decrease (TMDS) | MDE+TMDS             | MDE+TMDS             | TMDS                 |
| Hypomethylation  | DOWN      | No Change  | mRNA decrease (TPDS)         | Protein increase (TMDE) | TPDS+TMDE            | TPDS+TMDE            | TMDE                 |
| Hypomethylation  | No Change | No Change  | Methylation decrease (ncRNA) | None                    | MDE+ncRNA            | MDE_ncRNA            | MDE_ncRNA            |
| No Change        | DOWN      | UP         | mRNA decrease (TPDS)         | Protein increase (TMDE) | TMDE                 | TPDS+TMDE            | TMDE                 |
| No Change        | UP        | DOWN       | mRNA increase (TPDE)         | Protein decrease (TMDS) | TMDS                 | TPDE+TMDS            | TMDS                 |
| No Change        | DOWN      | DOWN       | mRNA decrease (TPDS)         | None                    | TPDS                 | TPDS                 | TPDS                 |
| No Change        | UP        | UP         | mRNA increase (TPDE)         | None                    | TPDE                 | TPDE                 | TPDE                 |
| No Change        | No Change | UP         | Protein increase (TMDE)      | None                    | TMDE                 | TMDE                 | TMDE                 |
| No Change        | No Change | DOWN       | Protein decrease (TMDS)      | None                    | TMDS                 | TMDS                 | TMDS                 |
| No Change        | UP        | No Change  | mRNA increase (TPDE)         | Protein decrease (TMDS) | TPDE+TMDS            | TPDE+TMDS            | TMDS                 |
| No Change        | DOWN      | No Change  | mRNA decrease (TPDS)         | Protein increase (TMDE) | TPDS+TMDE            | TPDS+TMDE            | TMDE                 |
| No Change        | No Change | No Change  | NoChange                     | NoChange                | NoChange             | NoChange             | NoChange             |

Sci-RCM also interfaces with scivae (the variational autoencoder component that we use to generate an integrated rank).
"""


class SciRCM:

    def __init__(self, meth_file: str,
                 rna_file: str,
                 proteomics_file: str,
                 rna_logfc: str,
                 rna_padj: str,
                 meth_diff: str,
                 meth_padj: str,
                 prot_logfc: str,
                 prot_padj: str,
                 gene_id: str,
                 rna_padj_cutoff=0.05,
                 prot_padj_cutoff=0.05,
                 meth_padj_cutoff=0.05,
                 rna_logfc_cutoff=1.0,
                 prot_logfc_cutoff=0.5,
                 meth_diff_cutoff=10,
                 output_dir='.',
                 output_filename=None,
                 non_coding_genes=None,
                 debug_on=False, sep=',',
                 bg_type='P|(M&R)',
                 sciutil=None,
                 logfile=None,
                 reg_grp_1_lbl='Regulation_Grouping_1',
                 reg_grp_2_lbl='Regulation_Grouping_2',
                 reg_grp_3_lbl='Regulation_Grouping_3',
                 main_reg_label='Regulation_Grouping_2'):
        self.u = SciUtil() if sciutil is None else sciutil
        plt.rcParams['svg.fonttype'] = 'none'
        self.meth_diff = meth_diff
        self.merged_df = None
        self.prot_logfc = prot_logfc
        self.rna_logfc = rna_logfc
        self.rna_padj = rna_padj
        self.meth_padj = meth_padj
        self.prot_padj = prot_padj
        self.gene_id = gene_id
        self.logfile = open(logfile, "w+") if logfile is not None else None  # File for logging results
        self.rna_padj_cutoff, self.meth_padj_cutoff, self.prot_padj_cutoff = rna_padj_cutoff, meth_padj_cutoff, prot_padj_cutoff
        self.rna_logfc_cutoff, self.meth_diff_cutoff, self.prot_logfc_cutoff = rna_logfc_cutoff, meth_diff_cutoff, prot_logfc_cutoff
        self.debug = debug_on
        self.output_dir = output_dir
        self.reg_grp_1_lbl = reg_grp_1_lbl
        self.reg_grp_2_lbl = reg_grp_2_lbl
        self.reg_grp_3_lbl = reg_grp_3_lbl
        self.main_reg_label = main_reg_label
        if main_reg_label != reg_grp_2_lbl and main_reg_label != reg_grp_1_lbl and main_reg_label != reg_grp_3_lbl:
            self.u.err_p([f'ERROR: your main regulatory label (main_reg_label) must be one of: '
                          f'{reg_grp_1_lbl}, {reg_grp_2_lbl} or {reg_grp_3_lbl}, you passed: ', main_reg_label])
        self.bg_type = bg_type
        # Otherwise this will be in the specific ones.
        if prot_logfc and rna_logfc and meth_diff:
            self.bg_list = ['(P&M)|(P&R)', '(P&M)|(P&R)|(M&R)', '*', 'P&M&R', 'P|M|R', 'P|(M&R)']
            if bg_type not in self.bg_list:
                self.u.err_p(['ERROR: selected background type was not allowed, please choose from one of: ', self.bg_list,
                              '\n Note: | means OR and & means AND'])
            self.output_filename = output_filename if output_filename else f'scircm_r{rna_logfc_cutoff}-{rna_padj_cutoff}' \
                                                                       f'_p{prot_logfc_cutoff}-{prot_padj_cutoff}' \
                                                                       f'_m{meth_diff_cutoff}-{meth_padj_cutoff}.csv'
            self.meth_df = pd.read_csv(meth_file, sep=sep)
            self.rna_df = pd.read_csv(rna_file, sep=sep)
            self.prot_df = pd.read_csv(proteomics_file, sep=sep)

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
        if len(self.meth_df[self.meth_df[self.gene_id].duplicated()]) > 0:
            num_dups = len(self.meth_df[self.meth_df[self.gene_id].duplicated()])
            self.u.warn_p(['Methylation dataset contained duplicates based on ID! Dropping duplicate IDs,'
                           ' note you should do this '
                           'before running SiRCle. We have just dropped it and kept the first entry. You had: ',
                           num_dups, 'duplicates.'])
            self.meth_df = self.meth_df[~self.meth_df[self.gene_id].duplicated(keep='first')]

        # Merge the dataframes together
        self.merge_dfs()
        # Generate background
        self.df = self.gen_bg()
        # Calculate groups
        self.run_rcm()
        # Save the DF and return the groupings
        self.df.to_csv(os.path.join(self.output_dir, self.output_filename), index=False)
        return self.df

    @staticmethod
    def _get_bg_filter(bg_type, prot_val, rna_val, meth_val, prot_padj_cutoff, meth_padj_cutoff, rna_padj_cutoff):
        """
        Filters the data based on a background selection. We have defined the filter such that it is based on being
        at minimum "significant". The gene doesn't have to ALSO meet the threshold (i.e. being above or less than
        a certain logFC).
        """
        c = 0
        if bg_type == 'P|(M&R)':  # Protein AND (DNA methylation OR RNA)
            c = 1 if prot_val <= prot_padj_cutoff else 0
            c += 1 if rna_val <= rna_padj_cutoff and meth_val <= meth_padj_cutoff else 0
        elif bg_type == 'P|M|R':  # Protein OR methylation OR RNA
            c = 1 if prot_val <= prot_padj_cutoff else 0
            c += 1 if rna_val <= rna_padj_cutoff else 0
            c += 1 if meth_val <= meth_padj_cutoff else 0
        elif bg_type == 'P&M&R':  # Protein AND Methylation AND RNA
            c = 1 if (prot_val <= prot_padj_cutoff and rna_val <= rna_padj_cutoff and meth_val <= meth_padj_cutoff) else 0
        elif bg_type == '(P&M)|(P&R)|(M&R)':  # At least two are significant
            c = 1 if (prot_val <= prot_padj_cutoff and rna_val <= rna_padj_cutoff) else 0
            c += 1 if (prot_val <= prot_padj_cutoff and meth_val <= meth_padj_cutoff) else 0
            c += 1 if (rna_val <= rna_padj_cutoff and meth_val <= meth_padj_cutoff) else 0
        elif bg_type == '(P&M)|(P&R)':  # Protein and one other
            c = 1 if (prot_val <= prot_padj_cutoff and rna_val <= rna_padj_cutoff) else 0
            c += 1 if (prot_val <= prot_padj_cutoff and meth_val <= meth_padj_cutoff) else 0
        elif bg_type == '*':  # Use all genes as the background
            c = 1
        return c

    def gen_bg(self, bg_type=None, prot_padj_cutoff=None, rna_padj_cutoff=None, meth_padj_cutoff=None):
        """
        Generate a background dataset i.e. since the RCM requires at least two of the 3 datasets to
        have a p value beneath a threshold we reduce our dataset to be smaller.
        """
        bg_type = bg_type if bg_type else self.bg_type
        prot_padj_cutoff = prot_padj_cutoff if prot_padj_cutoff else self.prot_padj_cutoff
        meth_padj_cutoff = meth_padj_cutoff if meth_padj_cutoff else self.meth_padj_cutoff
        rna_padj_cutoff = rna_padj_cutoff if rna_padj_cutoff else self.rna_padj_cutoff
        filter_vals = np.zeros(len(self.merged_df))
        meth_padj_values = self.merged_df[self.meth_padj].values
        prot_padj_values = self.merged_df[self.prot_padj].values

        # Choose the background dataframe
        for i, rna_padj in enumerate(self.merged_df[self.rna_padj].values):
            c = self._get_bg_filter(bg_type, prot_val=prot_padj_values[i], rna_val=rna_padj,
                                    meth_val=meth_padj_values[i],
                                    prot_padj_cutoff=prot_padj_cutoff, meth_padj_cutoff=meth_padj_cutoff,
                                    rna_padj_cutoff=rna_padj_cutoff)
            filter_vals[i] = c

        df = self.merged_df.copy()
        # Filter the df to become the background df
        df['Number of significant datasets'] = filter_vals
        # Filter DF to only include those that are sig in two.
        df = df[df['Number of significant datasets'] >= 1]
        df = df.reset_index()  # Reset the index of the dataframe

        return df

    def merge_dfs(self):
        """
        Merge the supplied RNAseq, DNA methylation and proteomics dataframes together. We do an outer join. This can
        result in gene identifiers becoming disjoint (i.e. if there are two ensembl ID columns, we get two now).
        """
        self.rna_df = self.rna_df.set_index(self.gene_id)
        self.df = self.rna_df.merge(self.meth_df, on=self.gene_id, how='outer', suffixes=['_r', '_m'])
        self.df = self.df.merge(self.prot_df, on=self.gene_id, how='outer', suffixes=['', '_p'])
        # Fill any of the values with NAs (p values need to be filled with 1's)
        self.df[[self.meth_padj]] = self.df[[self.meth_padj]].fillna(value=1.0)
        self.df[[self.rna_padj]] = self.df[[self.rna_padj]].fillna(value=1.0)
        self.df[[self.prot_padj]] = self.df[[self.prot_padj]].fillna(value=1.0)
        # Fill the rest of the values with 0's
        self.df = self.df.fillna(0)  # Fill any other values with 0 i.e. any of the logFCs or statistics.
        self.df.to_csv(os.path.join(self.output_dir, "SIRCLE_merged_df.csv"), index=False)
        self.merged_df = self.df.copy()

    def run_rcm(self):
        """
        Runs the regulatory clustering and assigns labels for the three levels of clustering:
        1) Level 1: which is the most granular
        2) Level 2: produces a medium level of filtering (this is the default)
        3) Level 3: produces a general idea of what the regulational level is on the basis of the protein.
        :return:
        """
        lbls = ['None'] * len(self.df)
        # Initialise the regulatory labels
        self.df[self.reg_grp_1_lbl] = lbls
        self.df[self.reg_grp_2_lbl] = lbls
        self.df[self.reg_grp_3_lbl] = lbls

        if self.logfile is not None:
            self.logfile.write(f'MethylationState,RNAseqState,ProteinState,Label,NumGenes\n')

        # -------------- Clusters with hyper-methylation
        # Hypermethylation | DOWN      | DOWN       | Methylation increase (MDS)   | None
        self.get_grp(meth_c='pos', rna_c='neg', prot_c='neg', grp_id='MDS', reg_grp_1='MDS', reg_grp_3='MDS')

        # Hypermethylation | UP        | DOWN       | mRNA increase (TPDE)         | Protein decrease (TMDS)
        self.get_grp(meth_c='pos', rna_c='pos', prot_c='neg', grp_id='TPDE_TMDS', reg_grp_1='TPDE_TMDS', reg_grp_3='TMDS')

        # Hypermethylation | UP        | UP         | mRNA increase (TPDE)         | None
        self.get_grp(meth_c='pos', rna_c='pos', prot_c='pos', grp_id='TPDE', reg_grp_1='TPDE', reg_grp_3='TPDE')

        # Hypermethylation | DOWN      | UP         | Methylation increase (MDS)   | Protein increase (TMDE)
        self.get_grp(meth_c='pos', rna_c='neg', prot_c='pos', grp_id='TMDE', reg_grp_1='MDS_TMDE', reg_grp_3='TMDE')

        # Hypermethylation | No Change | UP         | mRNA increase (TPDE)         | Protein increase (TMDE)
        self.get_grp(meth_c='pos', rna_c='-', prot_c='pos', grp_id='TMDE', reg_grp_1='TPDE_TMDE', reg_grp_3='TMDE')

        # Hypermethylation | No Change | DOWN       | mRNA increase (TPDE)         | Protein decrease (TMDS)
        self.get_grp(meth_c='pos', rna_c='-', prot_c='neg', grp_id='TMDS', reg_grp_1='TPDE_TMDS', reg_grp_3='TMDS')

        # Hypermethylation | UP        | No Change  | mRNA increase (TPDE)         | Protein decrease (TMDS)
        self.get_grp(meth_c='pos', rna_c='pos', prot_c='-', grp_id='TPDE_TMDS', reg_grp_1='TPDE_TMDS', reg_grp_3='TMDS')

        # Hypermethylation | DOWN      | No Change  | Methylation increase (MDS)   | Protein increase (TMDE)
        self.get_grp(meth_c='pos', rna_c='neg', prot_c='-', grp_id='MDS_TMDE', reg_grp_1='MDS_TMDE', reg_grp_3='TMDE')

        # -------------- Clusters with hypo-methylation
        # Hypomethylation  | DOWN      | DOWN       | mRNA decrease (TPDS)         | None
        self.get_grp(meth_c='neg', rna_c='neg', prot_c='neg', grp_id='TPDS', reg_grp_1='TPDS', reg_grp_3='TPDS')

        # Hypomethylation  | UP        | DOWN       | Methylation decrease (MDE)   | Protein decrease (TMDS)
        self.get_grp(meth_c='neg', rna_c='pos', prot_c='neg', grp_id='TMDS', reg_grp_1='MDE_TMDS', reg_grp_3='TMDS')

        # Hypomethylation  | UP        | UP         | Methylation decrease (MDE)   | None
        self.get_grp(meth_c='neg', rna_c='pos', prot_c='pos', grp_id='MDE', reg_grp_1='MDE', reg_grp_3='MDE')

        # Hypomethylation  | DOWN      | UP         | mRNA decrease (TPDS)         | Protein increase (TMDE)
        self.get_grp(meth_c='neg', rna_c='neg', prot_c='pos', grp_id='TPDS_TMDE', reg_grp_1='TPDS_TMDE', reg_grp_3='TMDE')

        # Hypomethylation  | No Change | UP         | mRNA decrease (TPDS)         | Protein increase (TMDE)
        self.get_grp(meth_c='neg', rna_c='-', prot_c='pos', grp_id='TMDE', reg_grp_1='TPDS_TMDE', reg_grp_3='TMDE')

        # Hypomethylation  | No Change | DOWN       | mRNA decrease (TPDS)         | Protein decrease (TMDS)
        self.get_grp(meth_c='neg', rna_c='-', prot_c='neg', grp_id='TMDS', reg_grp_1='TPDS_TMDS', reg_grp_3='TMDS')

        # Hypomethylation  | UP        | No Change  | Methylation decrease (MDE)   | Protein decrease (TMDS)
        self.get_grp(meth_c='neg', rna_c='pos', prot_c='-', grp_id='MDE_TMDS', reg_grp_1='MDE_TMDS', reg_grp_3='TMDS')

        # Hypomethylation  | DOWN      | No Change  | mRNA decrease (TPDS)         | Protein increase (TMDE)
        self.get_grp(meth_c='neg', rna_c='neg', prot_c='-', grp_id='TPDS_TMDE', reg_grp_1='TPDS_TMDE', reg_grp_3='TMDE')

        # -------------- Clusters with change in direction are mRNA and no DNA Methylation
        # No Change        | DOWN      | UP         | mRNA decrease (TPDS)         | Protein increase (TMDE)
        self.get_grp(meth_c='-', rna_c='neg', prot_c='pos', grp_id='TMDE', reg_grp_1='TPDS_TMDE', reg_grp_3='TMDE')

        # No Change        | UP        | DOWN       | mRNA increase (TPDE)         | Protein decrease (TMDS)
        self.get_grp(meth_c='-', rna_c='pos', prot_c='neg', grp_id='TMDS', reg_grp_1='TPDE_TMDS', reg_grp_3='TMDS')

        # -------------- Clusters with no methylation
        # No Change        | DOWN      | DOWN       | mRNA decrease (TPDS)         | None
        self.get_grp(meth_c='-', rna_c='neg', prot_c='neg', grp_id='TPDS',  reg_grp_1='TPDS', reg_grp_3='TPDS')

        # No Change        | UP        | UP         | mRNA increase (TPDE)         | None
        self.get_grp(meth_c='-', rna_c='pos', prot_c='pos', grp_id='TPDE',  reg_grp_1='TPDE', reg_grp_3='TPDE')

        # -------------- Clusters with only proteomics
        # No Change        | No Change | UP         | Protein increase (TMDE)      | None
        self.get_grp(meth_c='-', rna_c='-', prot_c='pos', grp_id='TMDE', reg_grp_1='TMDE', reg_grp_3='TMDE')

        # No Change        | No Change | DOWN       | Protein decrease (TMDS)      | None
        self.get_grp(meth_c='-', rna_c='-', prot_c='neg', grp_id='TMDS', reg_grp_1='TMDS', reg_grp_3='TMDS')

        # -------------- Clusters with only mRNA
        # No Change        | UP        | No Change  | mRNA increase (TPDE)         | Protein decrease (TMDS)
        self.get_grp(meth_c='-', rna_c='pos', prot_c='-', grp_id='TPDE_TMDS', reg_grp_1='TPDE_TMDS', reg_grp_3='TMDS')
        # No Change        | DOWN      | No Change  | mRNA decrease (TPDS)         | Protein increase (TMDE) | TPDS+TMDE
        self.get_grp(meth_c='-', rna_c='neg', prot_c='-', grp_id='TPDS_TMDE', reg_grp_1='TPDS_TMDE', reg_grp_3='TMDE')

        # -------------- Non-Coding gene clusters
        # Hypermethylation | No Change | No Change  | Methylation increase (ncRNA) | None
        self.get_grp(meth_c='pos', rna_c='-', prot_c='-', grp_id='MDS-ncRNA', reg_grp_1='MDS-ncRNA',
                     reg_grp_3='MDS-ncRNA', filter_list=self.non_coding_genes)
        # Hypomethylation  | No Change | No Change  | Methylation decrease (ncRNA) | None
        self.get_grp(meth_c='neg', rna_c='-', prot_c='-', grp_id='MDE-ncRNA', reg_grp_1='MDE-ncRNA',
                     reg_grp_3='MDE-ncRNA', filter_list=self.non_coding_genes)

        # Close the logfile
        if self.logfile is not None:
            self.logfile.close()
        return None

    def get_df(self):
        return self.df

    def integrate(self):
        # Here we combine the p-values and use a VAE to compute an integrated rank.
        combined_padj = []
        for i, p in enumerate(encoded_rcm['Protein_padj'].values):
            rna_p = encoded_rcm['RNA_padj'].values[i]
            cpg_p = encoded_rcm['CpG_padj'].values[i]
            combined_padj.append(combine_pvalues([p, rna_p, cpg_p])[1])
        encoded_rcm['combined_padj'] = combined_padj

    def get_grp(self, meth_c: str, rna_c: str, prot_c: str, grp_id: str, filter_list=None, reg_grp_1='', reg_grp_3=''):
        """ Compute all genes meeting a specific regulatory condition. """
        meth_change = self.df[self.meth_diff].values
        rna_change = self.df[self.rna_logfc].values
        prot_change = self.df[self.prot_logfc].values

        meth_padj = self.df[self.meth_padj].values
        rna_padj = self.df[self.rna_padj].values
        prot_padj = self.df[self.prot_padj].values

        grp = np.ones(len(meth_change))
        if rna_c == 'pos':
            grp *= 1.0 * (rna_change >= self.rna_logfc_cutoff) * (rna_padj <= self.rna_padj_cutoff)
        elif rna_c == 'neg':
            grp *= 1.0 * (rna_change <= (-1 * self.rna_logfc_cutoff)) * (rna_padj <= self.rna_padj_cutoff)
        else:
            # i.e. this gene should belong in the group if it fails to get assigned to the others i.e.
            # misses out based on the pvalue OR the logFC (hence the plus)
            grp *= (1.0 * (rna_padj > self.rna_padj_cutoff) + (1.0 * (abs(rna_change) < self.rna_logfc_cutoff)))

        if prot_c == 'pos':
            grp *= 1.0 * (prot_change >= self.prot_logfc_cutoff) * (1 * prot_padj <= self.prot_padj_cutoff)
        elif prot_c == 'neg':
            grp *= 1.0 * (prot_change <= (-1 * self.prot_logfc_cutoff)) * (1 * prot_padj <= self.prot_padj_cutoff)
        else:
            grp *= 1.0 * (1.0 * prot_padj > self.prot_padj_cutoff) + (abs(prot_change) < self.prot_logfc_cutoff)

        if meth_c == 'pos':
            grp *= 1.0 * (meth_change >= self.meth_diff_cutoff) * (meth_padj <= self.meth_padj_cutoff)
        elif meth_c == 'neg':
            grp *= 1.0 * (meth_change <= (-1 * self.meth_diff_cutoff)) * (meth_padj <= self.meth_padj_cutoff)
        else:
            grp *= 1.0 * ((1.0 * meth_padj > self.meth_padj_cutoff) + (abs(meth_change) < self.meth_diff_cutoff))

        # If we have a filter list of genes (e.g. for non-coding RNAs), then the genes MUST also be within this list
        genes_in_list = []
        if filter_list:
            for g in self.df[self.gene_id].values:
                if g in filter_list:
                    genes_in_list.append(1)
                else:
                    genes_in_list.append(0)
            grp *= np.array(genes_in_list)

        # Keep only the genes in this group
        grp_genes = self.df[self.gene_id].values[np.where(grp > 0)[0]]

        if self.debug:
            self.u.dp([grp_id, f'{meth_c} METH', f'{rna_c} RNA', f'{prot_c} PROT',
                  len(list(grp)), '\n', ', '.join(list(grp_genes))])
        if self.logfile is not None:
            # Print to logfile the results
            self.logfile.write(f'{meth_c},{rna_c},{prot_c},{grp_id},{len(list(grp_genes))}\n')
        # Add in the labels
        grp_ids = np.where(grp > 0)[0]

        # Create groups
        grp_1_labels = list(self.df[self.reg_grp_1_lbl].values)
        grp_2_labels = list(self.df[self.reg_grp_2_lbl].values)
        grp_3_labels = list(self.df[self.reg_grp_3_lbl].values)

        # Fill out the group labels 1 by 1 so that we can ensure that what we've added is correct.
        # Also if the user has set there to be overlaps (i.e. groups are not mutually exclusive) then this will print
        # out that there are possible duplicates.
        for i, g in enumerate(self.df[self.gene_id].values):
            if g in grp_genes:
                if grp_1_labels[i] and grp_1_labels[i] != 'None':
                    self.u.warn_p(["Possible duplicate Grouping 1:", grp_1_labels[i], reg_grp_1, g])
                    print(self.df[self.df[self.gene_id] == g])
                if grp_2_labels[i] and grp_2_labels[i] != 'None':
                    self.u.warn_p(["Possible duplicate Grouping 2:", grp_2_labels[i], grp_id, g])
                    print(self.df[self.df[self.gene_id] == g])
                if grp_1_labels[i] and grp_1_labels[i] != 'None':
                    self.u.warn_p(["Possible duplicate Grouping 3:", grp_3_labels[i], reg_grp_3, g])
                    print(self.df[self.df[self.gene_id] == g])
                grp_1_labels[i] = f'{reg_grp_1}_M-{meth_c}_R-{rna_c}_P-{prot_c}'
                grp_2_labels[i] = grp_id
                grp_3_labels[i] = reg_grp_3

        self.df[self.reg_grp_1_lbl] = grp_1_labels
        self.df[self.reg_grp_2_lbl] = grp_2_labels
        self.df[self.reg_grp_3_lbl] = grp_3_labels

        return self.df[self.gene_id][grp_ids]

    def save_df(self, output_filename):
        """ Save DF to a csv (mainly for R) """
        self.df.to_csv(output_filename, index=False)

    def get_genes_in_reg_grp(self, label, gene_label=None, reg_label=None):
        """ Get genes for a particular regulatory group. """
        reg_label = reg_label if reg_label else self.main_reg_label
        gene_label = self.gene_id if not gene_label else gene_label
        return list(set(self.df[self.df[reg_label] == label][gene_label].values))

    def get_all_assigned_genes(self, gene_label=None, reg_label=None):
        """ Get all genes assigned to a regulatory group (i.e. any that are not none) """
        reg_label = reg_label if reg_label else self.main_reg_label
        gene_label = self.gene_id if not gene_label else gene_label
        return list(set(self.df[self.df[reg_label] != 'None'][gene_label].values))

    def get_all_unassigned_genes(self, gene_label=None, reg_label=None):
        """ Get all genes that weren't assigned to a regulatory label. """
        gene_label = self.gene_id if not gene_label else gene_label
        reg_label = reg_label if reg_label else self.main_reg_label
        return list(set(self.df[self.df[reg_label] == 'None'][gene_label].values))


def filter_methylation_data_by_genes(cpg_df, gene_id, p_val, logfc):
    cpg_df_grped = cpg_df.groupby(gene_id)
    rows = []
    num_cpgs = []
    for cpg in cpg_df_grped:
        cpg = cpg[1]
        cpg = cpg[cpg[p_val] < 0.05]
        num_cpgs.append(len(cpg))
        if len(cpg) > 0:
            if len(cpg) < 3:
                add_row = True
            else:
                pos_cpg = cpg[cpg[logfc] > 0]
                neg_cpg = cpg[cpg[logfc] < 0]
                num_pos = len(pos_cpg)
                num_neg = len(neg_cpg)
                add_row = False
                if num_pos and num_pos/len(cpg) > 0.6:
                    cpg = pos_cpg
                    add_row = True
                elif num_neg and num_neg/len(cpg) > 0.6:
                    cpg = neg_cpg
                    add_row = True
            if add_row:
                max_cpg_idx = None
                max_t_value = 0  # absolute
                idxs = cpg.index
                for i, t in enumerate(cpg[logfc].values):
                    if abs(t) > abs(max_t_value):
                        max_t_value = t
                        max_cpg_idx = i
                rows.append(cpg[cpg.index == idxs[max_cpg_idx]].values[0])
    new_cpg_df = pd.DataFrame(rows, columns=cpg_df.columns)
    u = SciUtil()
    u.dp(['Originally had: ', len(cpg_df_grped), 'genes.\n', 'Filtered DF now has: ', len(new_cpg_df), ' genes.'])
    return new_cpg_df
