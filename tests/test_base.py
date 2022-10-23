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
from scircm import SciRCM, SciRCMnp
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
        rcm = SciRCM(self.meth_file, self.rna_file, self.prot_file, "logFC_r", "padj_r", "logFC_m", "padj_m",
                     "logFC_p", "padj_p", "gene_name", sep=',',  bg_type='(P&M)|(P&R)',
                     rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1, meth_diff_cutoff=10,
                     )
        rcm.run()
        # Read in the output file
        df = rcm.get_df()
        df.to_csv("test.csv")
        # Check the "label" column equals the reg label colum
        true_labels = df['label'].values
        for i, tst_label in enumerate(df['Regulation_Grouping_2'].values):
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                assert true_labels[i] == tst_label
            else:
                assert tst_label == "None"

    def test_rcm_simple(self):
        """ Test RCM that has all sig but misses out on the change."""

        rcm = SciRCM(f'{self.data_dir}/meth_rcm.csv', f'{self.data_dir}/rna_rcm.csv', f'{self.data_dir}/prot_rcm.csv',
                     "rna_logfc", "rna_padj", "meth_diff", "meth_padj",
                     "prot_logfc", "prot_padj", "gene_name", sep=',',  bg_type='(P&M)|(P&R)',
                     rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1, meth_diff_cutoff=10)
        rcm.run()
        # Read in the output file
        df = rcm.get_df()
        # Check the "label" column equals the reg label column
        rcm.u.dp(["Regulation grouping 2"])
        true_labels = df["Regulation_Grouping2_true"].values  # Need to add a true one
        genes = df['gene_name'].values
        for i, tst_label in enumerate(df[rcm.main_reg_label].values):  # The
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                print(genes[i], i, tst_label, true_labels[i])
                # Since we don't pass a non-coding gene list we need to ensure that it == None
                assert true_labels[i].replace('+', '_') == tst_label
            else:
                assert tst_label == "None"
        # Check regulation grouping 1
        rcm.u.dp(["Regulation grouping 1"])
        true_labels = df["Regulation_Grouping1_true"].values  # Need to add a true one
        genes = df['gene_name'].values
        for i, tst_label in enumerate(df[rcm.reg_grp_1_lbl].values):  # The
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                print(genes[i], i, tst_label, true_labels[i])
                # Since we don't pass a non-coding gene list we need to ensure that it == None
                if '+' in true_labels[i]:
                    assert true_labels[i].split('+')[0] == tst_label.split('_')[0]
                    assert true_labels[i].split('+')[1] == tst_label.split('_')[1]
                else:
                    assert true_labels[i].split('+')[0] == tst_label.split('_')[0]
                print(true_labels[i].replace('+', '_'), tst_label)
            else:
                assert tst_label == "None"

        # Check regulation grouping 3
        rcm.u.dp(["Regulation grouping 3"])
        true_labels = df["Regulation_Grouping3_true"].values  # Need to add a true one
        genes = df['gene_name'].values
        for i, tst_label in enumerate(df[rcm.reg_grp_3_lbl].values):  # The
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                print(genes[i], i, tst_label, true_labels[i])
                # Since we don't pass a non-coding gene list we need to ensure that it == None
                assert true_labels[i].replace('+', '_') == tst_label
            else:
                assert tst_label == "None"

    def test_nc_genes(self):
        nc_df = pd.read_csv(os.path.join(self.data_dir, 'hsapiens_go_non-coding_14122020.csv'))
        nc_genes = list(nc_df['Name'].values)
        rcm = SciRCM(self.meth_file, self.rna_file, self.prot_file, "logFC_r", "padj_r", "logFC_m", "padj_m",
                     "logFC_p", "padj_p", "gene_name", sep=',', bg_type='(P&M)|(P&R)',
                     rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1, meth_diff_cutoff=10,
                     non_coding_genes=nc_genes
                     )
        rcm.run()
        # Read in the output file
        df = rcm.get_df()
        # Check the "label" column equals the reg label colum
        true_labels = df['label'].values
        for i, tst_label in enumerate(df['Regulation_Grouping_2'].values):
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                assert true_labels[i] == tst_label
            elif tst_label and 'nc' in tst_label:
                assert tst_label == 'S-ncRNA'
            else:
                assert tst_label == "None"
        assigned = rcm.get_all_assigned_genes()
        unassigned = rcm.get_all_unassigned_genes()
        assert len(set(assigned + unassigned)) == len(assigned) + len(unassigned)
        assert len(assigned) + len(unassigned) == len(rcm.df)

    def test_bg(self):
        nc_df = pd.read_csv(os.path.join(self.data_dir, 'hsapiens_gene_ensembl-GRCh38.p13.csv'))
        nc_genes = list(nc_df['external_gene_name'].values)
        rcm = SciRCM(self.meth_file, self.rna_file, self.prot_file, "logFC_r", "padj_r", "logFC_m", "padj_m",
                     "logFC_p", "padj_p", "gene_name", sep=',',
                     rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1, meth_diff_cutoff=10,
                     non_coding_genes=nc_genes, bg_type='P|(M&R)'
                     )

        # Check protein or Methylation AND RNAseq work
        assert rcm._get_bg_filter('P|(M&R)', 0.001, 0.8, 0.001, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('P|(M&R)', 0.001, 0.001, 0.8, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('P|(M&R)', 0.1, 0.001, 0.001, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('P|(M&R)', 0.05, 0.001, 0.001, 0.05, 0.05, 0.05) == 2

        assert rcm._get_bg_filter('P|(M&R)', 0.1, 0.8, 0.001, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('P|(M&R)', 0.1, 0.001, 0.8, 0.05, 0.05, 0.05) == 0

        # Check P|M|R
        assert rcm._get_bg_filter('P|M|R', 0.001, 0.8, 0.1, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('P|M|R', 0.05, 0.8, 0.1, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('P|M|R', 0.5, 0.1, 0.04, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('P|M|R', 0.5, 0.04, 0.1, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('P|M|R', 0.5, 0.1, 0.1, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('P|M|R', 0.5, 0.04, 0.01, 0.05, 0.05, 0.05) == 2
        assert rcm._get_bg_filter('P|M|R', 0.05, 0.04, 0.01, 0.05, 0.05, 0.05) == 3

        # Check P&M&R
        assert rcm._get_bg_filter('P&M&R', 0.001, 0.8, 0.1, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('P&M&R', 0.001, 0.008, 0.1, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('P&M&R', 0.1, 0.008, 0.001, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('P&M&R', 0.001, 0.8, 0.001, 0.05, 0.05, 0.05) == 0

        assert rcm._get_bg_filter('P&M&R', 0.001, 0.008, 0.001, 0.05, 0.05, 0.05) == 1

        # Check (P&M)|(P&R)|(M&R)
        assert rcm._get_bg_filter('(P&M)|(P&R)|(M&R)', 0.1, 0.008, 0.1, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('(P&M)|(P&R)|(M&R)', 0.1, 0.1, 0.001, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('(P&M)|(P&R)|(M&R)', 0.001, 0.1, 0.1, 0.05, 0.05, 0.05) == 0

        assert rcm._get_bg_filter('(P&M)|(P&R)|(M&R)', 0.001, 0.008, 0.1, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('(P&M)|(P&R)|(M&R)', 0.1, 0.008, 0.01, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('(P&M)|(P&R)|(M&R)', 0.01, 0.1, 0.01, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('(P&M)|(P&R)|(M&R)', 0.01, 0.01, 0.01, 0.05, 0.05, 0.05) == 3

        # Check (P&M)|(P&R)
        assert rcm._get_bg_filter('(P&M)|(P&R)', 0.1, 0.01, 0.01, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('(P&M)|(P&R)', 0.01, 0.1, 0.1, 0.05, 0.05, 0.05) == 0

        assert rcm._get_bg_filter('(P&M)|(P&R)', 0.01, 0.01, 0.1, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('(P&M)|(P&R)', 0.05, 0.5, 0.01, 0.05, 0.05, 0.05) == 1

        assert rcm._get_bg_filter('(P&M)|(P&R)', 0.01, 0.01, 0.01, 0.05, 0.05, 0.05) == 2

        # Check *
        assert rcm._get_bg_filter('*', 1, 2, 3, 0.05, 0.05, 0.05) == 1

        # Check P&R # bg_type, prot_val, rna_val, meth_val, prot_padj_cutoff, meth_padj_cutoff, rna_padj_cutoff
        assert rcm._get_bg_filter('P&R', 1, 1, 0, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('P&R', 0, 0, 0, 0.05, 0.05, 0.05) == 1

        # Check P|R
        assert rcm._get_bg_filter('P|R', 1, 1, 0, 0.05, 0.05, 0.05) == 0
        assert rcm._get_bg_filter('P|R', 0, 0, 0, 0.05, 0.05, 0.05) == 2
        assert rcm._get_bg_filter('P|R', 0, 1, 0, 0.05, 0.05, 0.05) == 1
        assert rcm._get_bg_filter('P|R', 1, 0, 0, 0.05, 0.05, 0.05) == 1

    def test_bg_nc_gen(self):
        nc_df = pd.read_csv(os.path.join(self.data_dir, 'hsapiens_gene_ensembl-GRCh38.p13.csv'))
        meth_file = os.path.join(self.data_dir, 'meth_NC.csv')
        rna_file = os.path.join(self.data_dir, 'rna_NC.csv')
        prot_file = os.path.join(self.data_dir, 'prot_NC.csv')
        nc_genes = None #list(nc_df['external_gene_name'].values)
        rcm = SciRCM(meth_file, rna_file, prot_file, "logFC_r", "padj_r", "logFC_m", "padj_m",
                     "logFC_p", "padj_p", "gene_name", sep=',',
                     rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1, meth_diff_cutoff=10,
                     non_coding_genes=nc_genes, bg_type='P|(M&R)'
                     )
        # Check actually generating the bg
        rcm.merge_dfs()
        df = rcm.gen_bg()
        assert len(df) == 14

        nc_df = pd.read_csv(os.path.join(self.data_dir, 'hsapiens_gene_ensembl-GRCh38.p13.csv'))
        nc_genes = list(nc_df['external_gene_name'].values)
        rcm = SciRCM(meth_file, rna_file, prot_file, "logFC_r", "padj_r", "logFC_m", "padj_m",
                     "logFC_p", "padj_p", "gene_name", sep=',',
                     rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1, meth_diff_cutoff=10,
                     non_coding_genes=nc_genes, bg_type='P|(M&R)'
                     )
        # Check actually generating the bg
        rcm.merge_dfs()
        df = rcm.gen_bg()
        # i.e. that the three non-coding genes that are RNA only are added in!
        assert len(df) == 18
        # Now run the full way through!
        rcm = SciRCM(meth_file, rna_file, prot_file, "logFC_r", "padj_r", "logFC_m", "padj_m",
                     "logFC_p", "padj_p", "gene_name", sep=',',
                     rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
                     rna_logfc_cutoff=0.5, prot_logfc_cutoff=0.1, meth_diff_cutoff=10,
                     non_coding_genes=nc_genes, bg_type='P&M&R'
                     )
        df = rcm.run()
        # Check the "label" column equals the reg label colum
        true_labels = df['NC_label'].values
        gene_names = df['gene_name'].values
        for i, tst_label in enumerate(df['Regulation_Grouping_2'].values):
            if true_labels[i]:  # Otherwise we'd be testing between 0 and null
                assert true_labels[i] == tst_label
            else:
                print(gene_names[i])
                assert tst_label == "None"

        assigned = rcm.get_all_assigned_genes()
        unassigned = rcm.get_all_unassigned_genes()
        assert len(set(assigned + unassigned)) == len(assigned) + len(unassigned)
        assert len(assigned) + len(unassigned) == len(rcm.df)

