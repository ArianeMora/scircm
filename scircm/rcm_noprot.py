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
import os
from scircm import SciRCM
from sciutil import SciException
import pandas as pd


class SciRCMException(SciException):
    def __init__(self, message=''):
        Exception.__init__(self, message)


class SciRCMnp(SciRCM):

    def __init__(self, meth_file: str, rna_file: str, rna_logfc: str, rna_padj: str, meth_diff: str, meth_padj: str,
                 gene_id: str, rna_padj_cutoff=0.05,
                 prot_padj_cutoff=0.05, meth_padj_cutoff=0.05, rna_logfc_cutoff=1.0, prot_logfc_cutoff=0.5,
                 meth_diff_cutoff=10, output_dir='.', output_filename=None, non_coding_genes=None, debug_on=False,
                 sep=',', bg_type='M|R', sciutil=None, logfile=None,
                 reg_grp_1_lbl='Regulation_Grouping_1',
                 reg_grp_2_lbl='Regulation_Grouping_2',
                 reg_grp_3_lbl='Regulation_Grouping_3',
                 main_reg_label='Regulation_Grouping_2'):
        super().__init__(meth_file, rna_file, None, rna_logfc, rna_padj, meth_diff, meth_padj, None, None,
                         gene_id,
                         rna_padj_cutoff, prot_padj_cutoff, meth_padj_cutoff,
                         rna_logfc_cutoff, prot_logfc_cutoff, meth_diff_cutoff, output_dir,
                         output_filename, non_coding_genes, debug_on, sep,
                         bg_type, sciutil, logfile, reg_grp_1_lbl=reg_grp_1_lbl,
                         reg_grp_2_lbl=reg_grp_2_lbl, reg_grp_3_lbl=reg_grp_3_lbl, main_reg_label=main_reg_label)
        self.bg_type = bg_type
        self.bg_list = ['M&R', 'M|R', '*']
        if bg_type not in self.bg_list:
            self.u.err_p(['ERROR: selected background type was not allowed, please choose from one of: ', self.bg_list,
                          '\n Note: | means OR and & means AND'])
        self.output_filename = output_filename if output_filename else f'scircm_r{rna_logfc_cutoff}-{rna_padj_cutoff}' \
                                                                       f'_m{meth_diff_cutoff}-{meth_padj_cutoff}.csv'
        self.meth_df = pd.read_csv(meth_file, sep=sep)
        self.rna_df = pd.read_csv(rna_file, sep=sep)
        self.non_coding_genes = non_coding_genes
        # Contains genes for the non-coding region (use for human only).
        self.df = None
        # Contains the vae data
        self.vae = None

    @staticmethod
    def _get_bg_filter(bg_type, rna_val, meth_val, meth_padj_cutoff, rna_padj_cutoff):
        c = 0
        if bg_type == 'M&R':  # Protein AND (DNA methylation OR RNA)
            c = 1 if rna_val <= rna_padj_cutoff and meth_val <= meth_padj_cutoff else 0
        elif bg_type == 'M|R':  # Protein OR methylation OR RNA
            c = 1 if rna_val <= rna_padj_cutoff else 0
            c += 1 if meth_val <= meth_padj_cutoff else 0
        elif bg_type == '*':  # Use all genes as the background
            c = 1
        return c

    def gen_bg(self, bg_type=None, rna_padj_cutoff=None, meth_padj_cutoff=None):
        """
        Generate a background dataset i.e. since the RCM requires at least two of the 3 datasets to
        have a p value beneath a threshold we reduce our dataset to be smaller.
        """
        bg_type = bg_type if bg_type else self.bg_type
        meth_padj_cutoff = meth_padj_cutoff if meth_padj_cutoff else self.meth_padj_cutoff
        rna_padj_cutoff = rna_padj_cutoff if rna_padj_cutoff else self.rna_padj_cutoff
        filter_vals = np.zeros(len(self.merged_df))
        meth_padj_values = self.merged_df[self.meth_padj].values

        # Choose the background dataframe
        for i, rna_padj in enumerate(self.merged_df[self.rna_padj].values):
            c = self._get_bg_filter(bg_type, rna_padj, meth_padj_values[i],
                                    meth_padj_cutoff, rna_padj_cutoff)
            filter_vals[i] = c

        df = self.merged_df.copy()
        # Filter the df to become the background df
        df['Number of significant datasets'] = filter_vals
        # Filter DF to only include those that are sig in two.
        df = df[df['Number of significant datasets'] >= 1]
        df = df.reset_index()  # Reset the index of the dataframe

        return df

    def merge_dfs(self, fill_protein=False, protein_cols=None):
        self.rna_df = self.rna_df.set_index(self.gene_id)
        self.df = self.rna_df.merge(self.meth_df, on=self.gene_id, how='outer', suffixes=['_r', '_m'])
        # Fill any of the values with NAs (p values need to be filled with 1's)
        self.df[[self.meth_padj]] = self.df[[self.meth_padj]].fillna(value=1.0)
        self.df[[self.rna_padj]] = self.df[[self.rna_padj]].fillna(value=1.0)
        # Fill the rest of the values with 0's
        self.df = self.df.fillna(0)
        self.df.to_csv(os.path.join(self.output_dir, "merged_df.csv"), index=False)
        self.merged_df = self.df.copy()

    def run_rcm(self):
        lbls = ['None'] * len(self.df)
        # Initialise the regulatory labels
        self.df[self.reg_grp_1_lbl] = lbls
        self.df[self.reg_grp_2_lbl] = lbls
        self.df[self.reg_grp_3_lbl] = lbls
        if self.logfile is not None:
            self.logfile.write(f'MethylationState,RNAseqState,Label,NumGenes\n')

        # -------------- Clusters with only mRNA
        # No Change        | UP        | No Change  | mRNA increase (TPDE)
        self.get_grp(meth_c='-', rna_c='pos', grp_id='TPDE', reg_grp_1='TPDE', reg_grp_3='TPDE')
        # No Change        | DOWN      | No Change  | mRNA decrease (TPDS)
        self.get_grp(meth_c='-', rna_c='neg', grp_id='TPDS', reg_grp_1='TPDS', reg_grp_3='TPDS')

        # -------------- Clusters with both mRNA and DNA Methylation
        # Hypomethylation  | UP        | No Change  | Methylation decrease (MDE)
        self.get_grp(meth_c='neg', rna_c='pos', grp_id='MDE', reg_grp_1='MDE', reg_grp_3='MDE')

        # Hypomethylation  | DOWN      | No Change  | mRNA decrease (TPDS)
        self.get_grp(meth_c='neg', rna_c='neg', grp_id='TPDS', reg_grp_1='TPDS', reg_grp_3='TPDS')

        # Hypermethylation  | UP        | No Change       | Methylation decrease (MDE)
        self.get_grp(meth_c='pos', rna_c='pos', grp_id='TPDE', reg_grp_1='TPDE', reg_grp_3='TPDE')

        # Hypermethylation | DOWN      | DOWN       | Methylation increase (MDS)   | None
        self.get_grp(meth_c='pos', rna_c='neg', grp_id='MDS', reg_grp_1='MDS', reg_grp_3='MDS')

        # Here we want to add Methylation driven genes
        # -------------- Non-Coding gene clusters
        # Hypermethylation | No Change | No Change  | Methylation increase (ncRNA) | None
        self.get_grp(meth_c='pos', rna_c='-', grp_id='MDS-ncRNA', reg_grp_1='MDS-ncRNA',
                     reg_grp_3='MDS-ncRNA', filter_list=self.non_coding_genes)
        # Hypomethylation  | No Change | No Change  | Methylation decrease (ncRNA) | None
        self.get_grp(meth_c='neg', rna_c='-', grp_id='MDE-ncRNA', reg_grp_1='MDE-ncRNA',
                     reg_grp_3='MDE-ncRNA', filter_list=self.non_coding_genes)

        # Close the logfile
        if self.logfile is not None:
            self.logfile.close()

    def get_df(self):
        return self.df

    def get_grp(self, meth_c, rna_c, grp_id, reg_grp_1, reg_grp_3, filter_list=None):
        """ Get a single group """
        meth_change = self.df[self.meth_diff].values
        rna_change = self.df[self.rna_logfc].values

        meth_padj = self.df[self.meth_padj].values
        rna_padj = self.df[self.rna_padj].values

        grp = np.ones(len(meth_change))
        if rna_c == 'pos':
            grp *= 1.0 * (rna_change >= self.rna_logfc_cutoff) * (rna_padj <= self.rna_padj_cutoff)
        elif rna_c == 'neg':
            grp *= 1.0 * (rna_change <= (-1 * self.rna_logfc_cutoff)) * (rna_padj <= self.rna_padj_cutoff)
        else:
            # i.e. this gene should belong in the group if it fails to get assigned to the others i.e.
            # misses out based on the pvalue OR the logFC (hence the plus)
            grp *= (1.0 * (rna_padj > self.rna_padj_cutoff) + (1.0 * (abs(rna_change) < self.rna_logfc_cutoff)))

        if meth_c == 'pos':
            grp *= 1.0 * (meth_change >= self.meth_diff_cutoff) * (meth_padj <= self.meth_padj_cutoff)
        elif meth_c == 'neg':
            grp *= 1.0 * (meth_change <= (-1 * self.meth_diff_cutoff)) * (meth_padj <= self.meth_padj_cutoff)
        else:
            grp *= 1.0 * ((1.0 * meth_padj > self.meth_padj_cutoff) + (abs(meth_change) < self.meth_diff_cutoff))

        # If we have a filter list of genes, then the genes MUST also be within this list
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
            self.u.dp([grp_id, f'{meth_c} METH', f'{rna_c} RNA',
                  len(list(grp)), '\n', ', '.join(list(grp_genes))])
        if self.logfile is not None:
            # Print to logfile the results
            self.logfile.write(f'{meth_c},{rna_c},{grp_id},{len(list(grp_genes))}\n')
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
                grp_1_labels[i] = reg_grp_1
                grp_2_labels[i] = grp_id
                grp_3_labels[i] = reg_grp_3

        self.df[self.reg_grp_1_lbl] = grp_1_labels
        self.df[self.reg_grp_2_lbl] = grp_2_labels
        self.df[self.reg_grp_3_lbl] = grp_3_labels

        return self.df[self.gene_id][grp_ids]