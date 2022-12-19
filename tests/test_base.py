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

import pandas as pd
import os
import shutil
import tempfile
import unittest
from scircm import SciRCM, SciRP, SciMR
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class TestClass(unittest.TestCase):

    @classmethod
    def setup_class(self):
        local = True
        # Create a base object since it will be the same for all the tests
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data_dir = os.path.join(THIS_DIR, 'data')
        if local:
            self.tmp_dir = os.path.join(THIS_DIR, 'data', 'tmp')
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.mkdir(self.tmp_dir)
        else:
            self.tmp_dir = tempfile.mkdtemp(prefix='scircm_tmp_')
        # Setup the default data for each of the tests
        self.meth_file = os.path.join(self.data_dir, 'meth.csv')
        self.rna_file = os.path.join(self.data_dir, 'rna.csv')
        self.prot_file = os.path.join(self.data_dir, 'prot.csv')

        self.hg38_annot = os.path.join(self.data_dir, 'hsapiens_gene_ensembl-GRCh38.p13.csv')

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)


class TestSciRCM(TestClass):

    def test_base(self):
        """
        self, meth_file: str, rna_file: str, proteomics_file: str,
                 rna_logfc: str, rna_padj: str,
                 meth_diff: str, meth_padj: str,
                 prot_logfc: str, prot_padj: str,
                 rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
                 rna_logfc_cutoff=1.0, prot_logfc_cutoff=0.5, meth_diff_cutoff=10, output_dir='.',
                 output_filename=None,
                 debug_on=False,
                 gene_id=None,
                 reg_grp_1_lbl='Regulation_Grouping_1', reg_grp_2_lbl='Regulation_Grouping_2',
                 reg_grp_3_lbl='Regulation_Grouping_3',
                 main_reg_label='Regulation_Grouping_2'
        """
        rcm_file = 'data/FINAL_TABLE.csv.csv'
        rcm = SciRCM('data/FINAL_TABLE_CpG.csv', 'data/FINAL_TABLE_RNA.csv', 'data/FINAL_TABLE_Protein.csv',
                     "rna_logfc", "rna_padj", "meth_diff", "meth_padj",
                     "protein_logfc", "protein_padj", "gene_name", sep=',',  bg_type='(P&M)|(P&R)',
                     rna_padj_cutoff=0.1, prot_padj_cutoff=0.1, meth_padj_cutoff=0.1,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1, meth_diff_cutoff=0.1,
                     )
        rcm.run()
        # Read in the output file
        df = rcm.get_df()
        print(df.columns)
        # Check the "label" column equals the reg label colum
        true_labels = df['Regulation Grouping Changes (RG2)'].values
        genes = df['gene_name'].values
        for i, tst_label in enumerate(df['RG2_Changes'].values):
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                print(genes[i])
                assert true_labels[i].strip().replace('+', '_') == tst_label
            else:
                print(genes[i])
                assert tst_label == "None"

    def test_RNA_protein(self):
        rcm = SciRP('data/FINAL_TABLE_RNA_RNA-Protein.csv', 'data/RNA_Protein_ONLY_Final_table.csv',
                     "rna_logfc", "rna_padj",
                     "protein_logfc", "protein_padj", "gene_name", sep=',',  bg_type='*',
                     rna_padj_cutoff=0.1, prot_padj_cutoff=0.1,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1,
                     )
        rcm.run()
        # Read in the output file
        df = rcm.get_df()
        print(df.columns)
        # Check the "label" column equals the reg label colum
        true_labels = df['Regulation Grouping Changes (RG2)'].values
        genes = df['gene_name'].values
        for i, tst_label in enumerate(df['RG2_Changes'].values):
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                print(genes[i])
                assert true_labels[i].strip().replace('+', '_') == tst_label
            else:
                print(genes[i])
                assert tst_label == "None"

    def test_RNA_Methylation(self):
        rcm = SciMR('data/FINAL_TABLE_MethylationRNA_Meth.csv', 'data/FINAL_TABLE_MethylationRNA_RNA.csv',
                     "rna_logfc", "rna_padj", "meth_diff", "meth_padj",
                     "gene_name", sep=',', bg_type='M|R',
                     rna_padj_cutoff=0.1, meth_padj_cutoff=0.1,
                     rna_logfc_cutoff=0.5, meth_diff_cutoff=0.1)
        rcm.run()
        # Read in the output file
        df = rcm.get_df()
        print(df.columns)
        # Check the "label" column equals the reg label colum
        true_labels = df['Regulation Grouping Changes (RG2)_m'].values
        genes = df['gene_name'].values
        for i, tst_label in enumerate(df['RG2_Changes'].values):
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                print(genes[i], true_labels[i].strip().replace('+', '_'), tst_label)
                assert true_labels[i].strip().replace('+', '_') == tst_label
            else:
                print(genes[i])
                assert tst_label == "None"