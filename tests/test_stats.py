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

import os
import unittest

import pandas as pd

from scircm import RCMStats
import seaborn as sns
from sciviso import Heatmap
import matplotlib.pyplot as plt

class TestClass(unittest.TestCase):

    @classmethod
    def setup_class(self):
        local = True
        # Create a base object since it will be the same for all the tests
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data_dir = os.path.join(THIS_DIR, 'data')
        # if local:
        #     self.tmp_dir = os.path.join(THIS_DIR, 'data', 'tmp')
        #     if os.path.exists(self.tmp_dir):
        #         shutil.rmtree(self.tmp_dir)
        #     os.mkdir(self.tmp_dir)
        # else:
        #     self.tmp_dir = tempfile.mkdtemp(prefix='scircm_tmp_')
        # # Setup the default data for each of the tests
        self.meth_file = os.path.join(self.data_dir, 'meth.csv')
        self.rna_file = os.path.join(self.data_dir, 'rna.csv')
        self.prot_file = os.path.join(self.data_dir, 'prot.csv')

        self.hg38_annot = os.path.join(self.data_dir, 'hsapiens_gene_ensembl-GRCh38.p13.csv')

    @classmethod
    def teardown_class(self):
        print("Done")
        #shutil.rmtree(self.tmp_dir)


class TestRCMVAE(TestClass):


    def test_lin_vae(self):
        data_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/vae_runs/mds/'
        meth_file = f'{data_dir}MDS_data_cpg.csv'
        rna_file = f'{data_dir}MDS_data_rna.csv'
        protein_file = f'{data_dir}MDS_data_protein.csv'

        meth_sample_file = f'{data_dir}MDS_sample_cpg.csv'
        rna_sample_file = f'{data_dir}MDS_sample_rna.csv'
        protein_sample_file = f'{data_dir}MDS_sample_protein.csv'

        epochs = 50  # To make it quicker to train
        batch_size = 16
        num_nodes = 6
        mmd_weight = 1.0
        loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': mmd_weight}
        config = {"loss": loss,
                  "encoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "decoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "latent": {"num_nodes": 1},
                  "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                  "epochs": epochs,
                  "batch_size": batch_size
                  }
        sv = RCMVAE(os.path.join(self.data_dir, 'MDS_RCM.csv'),
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=data_dir,
                    stat_condition_column='stat_condition_label',
                    sub_condition_id_column='stat_condition_id',
                    run_name=f'nodes-{num_nodes}', normalise='rows')
        sv.train_vae(config=config)
        sv.run_vae_stats()

        sv = RCMVAE(os.path.join(self.data_dir, 'MDS_RCM.csv'),
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=data_dir,
                    stat_condition_column='stat_condition_label',
                    sub_condition_id_column='stat_condition_id',
                    run_name=f'nodes-{num_nodes}', normalise='columns')
        sv.train_vae(config=config)
        sv.run_vae_stats()

        # Test running linear

    def test_diff_config(self):
        data_dir = 'data/'
        meth_file = f'{data_dir}MDS_data_cpg.csv'
        rna_file = f'{data_dir}MDS_data_rna.csv'
        protein_file = f'{data_dir}MDS_data_protein.csv'

        meth_sample_file = f'{data_dir}MDS_sample_cpg.csv'
        rna_sample_file = f'{data_dir}MDS_sample_rna.csv'
        protein_sample_file = f'{data_dir}MDS_sample_protein.csv'

        loss = {'loss_type': 'multi', 'distance_metric': 'mmd', 'mmd_weight': 0.1, 'multi_loss': ['mse', 'mse']}
        epochs = 50  # To make it quicker to train
        batch_size = 16
        num_nodes = 6
        for r in range(0, 5):
            config = {"loss": loss,
                      "encoding": {"layers": [[{"num_nodes": num_nodes, "activation_fn": "relu"},
                                               {"num_nodes": num_nodes, "activation_fn": "relu"}]]},
                      "decoding": {"layers": [[{"num_nodes": num_nodes, "activation_fn": "relu"},
                                               {"num_nodes": num_nodes, "activation_fn": "relu"}]]},
                      "latent": {"num_nodes": 1},
                      "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                      "input_size": [3, 4],
                      "output_size": [3, 4],
                      "epochs": epochs,
                      "batch_size": batch_size
                      }
            sv = RCMVAE(os.path.join(self.data_dir, 'MDS_RCM.csv'),
                        meth_file, meth_sample_file, rna_file, rna_sample_file,
                        protein_file, protein_sample_file, output_folder=data_dir,
                        stat_condition_column='stat_condition_label',
                        sub_condition_id_column='stat_condition_id',

                        run_name=f'nodes-{num_nodes}_{r}')
            sv.train_vae(config=config)
            sv.run_vae_stats()

    def test_final_multi(self):
        data_dir = 'data/'

        output_dir = '/Users/ariane/Documents/code/sircle_meth/data/vae_2_multi_columns/'
        rcm_file = f'{data_dir}CPTAC_RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        meth_file = f'{data_dir}CPTAC_data_cpg.csv'
        rna_file = f'{data_dir}CPTAC_data_rna.csv'
        protein_file = f'{data_dir}CPTAC_data_protein.csv'

        meth_sample_file = f'{data_dir}CPTAC_sample_cpg.csv'
        rna_sample_file = f'{data_dir}CPTAC_sample_rna.csv'
        protein_sample_file = f'{data_dir}CPTAC_sample_protein.csv'

        epochs = 100  # To make it quicker to train
        batch_size = 16
        num_nodes = 2
        mmd_weight = 0.1
        loss = {'loss_type': 'multi', 'distance_metric': 'mmd', 'mmd_weight': mmd_weight, 'multi_loss': ['mse', 'mse']}
        config = {"loss": loss,
                  "encoding": {"layers": [[{"num_nodes": num_nodes, "activation_fn": "relu"},
                                           {"num_nodes": num_nodes, "activation_fn": "relu"}]]},
                  "decoding": {"layers": [[{"num_nodes": num_nodes, "activation_fn": "relu"},
                                           {"num_nodes": num_nodes, "activation_fn": "relu"}]]},
                  "latent": {"num_nodes": 1},
                  "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                  "input_size": [3, 4],
                  "output_size": [3, 4],
                  "epochs": epochs,
                  "batch_size": batch_size
                  }
        norm_type = 'columns'
        run_type = 'multi'
        rep = '1'
        sv = RCMVAE(rcm_file,
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=output_dir,
                    stat_condition_column='stat_condition_label',
                    sub_condition_id_column='stat_condition_id',
                    run_name=f'{num_nodes}-{rep}-{run_type}-{mmd_weight}-{norm_type}', normalise=norm_type)
        sv.train_vae(config=config)
        sv.run_vae_stats()


    def test_final(self):
        data_dir = 'data/'

        output_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/trained_vae/'
        rcm_file = f'{data_dir}CPTAC_RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        meth_file = f'{data_dir}CPTAC_data_cpg.csv'
        rna_file = f'{data_dir}CPTAC_data_rna.csv'
        protein_file = f'{data_dir}CPTAC_data_protein.csv'

        meth_sample_file = f'{data_dir}CPTAC_sample_cpg.csv'
        rna_sample_file = f'{data_dir}CPTAC_sample_rna.csv'
        protein_sample_file = f'{data_dir}CPTAC_sample_protein.csv'

        epochs = 100  # To make it quicker to train
        batch_size = 16
        num_nodes = 5
        mmd_weight = 0.1
        loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': mmd_weight}
        config = {"loss": loss,
                  "encoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "decoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "latent": {"num_nodes": 1},
                  "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                  "epochs": epochs,
                  "batch_size": batch_size
        }
        norm_type = 'rows'
        run_type = 'lin'
        rep = '1'
        sv = RCMVAE(rcm_file,
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=output_dir,
                    stat_condition_column='stat_condition_label',
                    sub_condition_id_column='stat_condition_id',
                    run_name=f'{num_nodes}-{rep}-{run_type}-{mmd_weight}-{norm_type}', normalise=norm_type)
        sv.train_vae(config=config)
        #sv.run_vae_stats()

    def test_s4(self):
        data_dir = 'data/'
        data_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/trained_vae/'

        output_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/trained_vae/'
        rcm_file = f'{data_dir}CPTAC_RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        cond0 = 'Stage1'
        cond1 = 'Stage4'

        meth_file = f'{data_dir}CPTAC_data_cpg_{cond0}-{cond1}.csv'
        rna_file = f'{data_dir}CPTAC_data_rna_{cond0}-{cond1}.csv'
        protein_file = f'{data_dir}CPTAC_data_protein_{cond0}-{cond1}.csv'

        meth_sample_file = f'{data_dir}CPTAC_sample_cpg_{cond0}-{cond1}.csv'
        rna_sample_file = f'{data_dir}CPTAC_sample_rna_{cond0}-{cond1}.csv'
        protein_sample_file = f'{data_dir}CPTAC_sample_protein_{cond0}-{cond1}.csv'

        epochs = 100  # To make it quicker to train
        batch_size = 16
        num_nodes = 5
        mmd_weight = 0.1
        loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': mmd_weight}
        config = {"loss": loss,
                  "encoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "decoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "latent": {"num_nodes": 1},
                  "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                  "epochs": epochs,
                  "batch_size": batch_size
        }
        norm_type = 'columns'
        run_type = 'lin'
        rep = '1'
        sv = RCMVAE(rcm_file,
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=output_dir,
                    stat_condition_column='stat_condition_label',
                    sub_condition_id_column='stat_condition_id',
                    run_name=f'{num_nodes}-{rep}-{run_type}-{mmd_weight}-{norm_type}', normalise=norm_type)
        sv.train_vae(config=config)
        sv.run_vae_stats()

    def test_diff_input(self):
        # scale_data
        data_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/VAE_final_rows_cols_0.1/input_data/'

        output_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/VAE_final_rows_cols_0.1/'
        rcm_file = f'{data_dir}CPTAC_RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        cond0 = 'Stage1'
        cond_runs = ['Stage4', 'Stage3', 'Stage2']

        for cond1 in cond_runs:
            meth_file = f'{data_dir}CPTAC_data_cpg.csv'
            rna_file = f'{data_dir}CPTAC_data_rna.csv'
            protein_file = f'{data_dir}CPTAC_data_protein.csv'

            meth_sample_file = f'{data_dir}CPTAC_sample_cpg_{cond0}-{cond1}.csv'
            rna_sample_file = f'{data_dir}CPTAC_sample_rna_{cond0}-{cond1}.csv'
            protein_sample_file = f'{data_dir}CPTAC_sample_protein_{cond0}-{cond1}.csv'

            epochs = 200  # To make it quicker to train
            batch_size = 16
            num_nodes = 5
            mmd_weight = 0.25
            loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': mmd_weight}
            config = {"loss": loss,
                      "encoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                      "decoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                      "latent": {"num_nodes": 1},
                      "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                      "epochs": epochs,
                      "batch_size": batch_size
            }
            norm_type = 'rows'
            run_type = 'lin'
            rep = '1'
            num_nodes = 5
            mmd_weight = 0.25
            sv = RCMVAE(rcm_file,
                        meth_file, meth_sample_file, rna_file, rna_sample_file,
                        protein_file, protein_sample_file, output_folder=output_dir,
                        stat_condition_column='stat_condition_label',
                        sub_condition_id_column='stat_condition_id',
                        run_name=f'{num_nodes}-{rep}-{run_type}-{mmd_weight}-{norm_type}', normalise=norm_type)
            if cond1 == 'Stage4':  # only train once
                sv.train_vae(config=config)
            sv.run_vae_stats(f'{cond0}vs{cond1}')


    def test_stats(self):
        data_dir = 'data/'
        # meth_file = f'{data_dir}MDS_meth_rcm.csv'
        # rna_file = f'{data_dir}MDS_rna_rcm.csv'
        # protein_file = f'{data_dir}MDS_protein_rcm.csv'
        #
        # rcm = SciRCM(meth_file, rna_file, protein_file, "logFC_rna", "padj_rna", "CpG_Beta_diff", "padj_meth",
        #              "logFC_protein", "padj_protein", "external_gene_name", sep=',',  bg_type='(P&M)|(P&R)',
        #              rna_padj_cutoff=0.05, prot_padj_cutoff=0.05, meth_padj_cutoff=0.05,
        #              rna_logfc_cutoff=1.0, proxt_logfc_cutoff=0.5, meth_diff_cutoff=0.1,
        #              )
        # rcm.run()
        # # Read in the output file
        # df = rcm.get_df()
        # # Let's try testing the VAE stats
        # df.set_index('external_gene_name', inplace=True)
        # df.to_csv(os.path.join(self.data_dir, 'RCM_MDS.csv'), index=True)
        # Now we want to check stats where we use the data that accomnaied those inputs
        meth_file = f'{data_dir}MDS_data_cpg.csv'
        rna_file = f'{data_dir}MDS_data_rna.csv'
        protein_file = f'{data_dir}MDS_data_protein.csv'

        meth_sample_file = f'{data_dir}MDS_sample_cpg.csv'
        rna_sample_file = f'{data_dir}MDS_sample_rna.csv'
        protein_sample_file = f'{data_dir}MDS_sample_protein.csv'
        sv = RCMVAE(os.path.join(self.data_dir, 'MDS_RCM.csv'),
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=data_dir,
                    stat_condition_column='stat_condition_label', sub_condition_id_column='stat_condition_id')
        sv.train_vae()
        sv.run_vae_stats()

    def test_stats_only(self):
        data_dir = 'data/'
        # Now we want to check stats where we use the data that accomnaied those inputs
        meth_file = f'{data_dir}MDS_data_cpg.csv'
        rna_file = f'{data_dir}MDS_data_rna.csv'
        protein_file = f'{data_dir}MDS_data_protein.csv'

        meth_sample_file = f'{data_dir}MDS_sample_cpg.csv'
        rna_sample_file = f'{data_dir}MDS_sample_rna.csv'
        protein_sample_file = f'{data_dir}MDS_sample_protein.csv'
        sv = RCMVAE(os.path.join(self.data_dir, 'MDS_RCM.csv'),
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=data_dir,
                    stat_condition_column='stat_condition_label', sub_condition_id_column='stat_condition_id')
        sv.run_vae_stats()

    def test_stats_full(self):
        data_dir = 'data/'
        rcm_file = f'{data_dir}CPTAC_RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        meth_file = f'{data_dir}CPTAC_data_cpg.csv'
        rna_file = f'{data_dir}CPTAC_data_rna.csv'
        protein_file = f'{data_dir}CPTAC_data_protein.csv'

        meth_sample_file = f'{data_dir}CPTAC_sample_cpg.csv'
        rna_sample_file = f'{data_dir}CPTAC_sample_rna.csv'
        protein_sample_file = f'{data_dir}CPTAC_sample_protein.csv'
        sv = RCMVAE(rcm_file,
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file,
                    output_folder="/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/VAE_tests/vae_new_norm/",
                    stat_condition_column='stat_condition_label', sub_condition_id_column='stat_condition_id',
                    run_name='CPTAC')
        sv.train_vae()
        sv.run_vae_stats()

    def test_lin_decoding(self):
        data_dir = 'data/'

        base_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/VAE/'
        output_dir = f'{base_dir}output_data/stats_files/'
        f_output_dir = f'{base_dir}output_data/enc_dec/'
        data_dir = f'{base_dir}input_data/'
        rcm_file = f'{data_dir}CPTAC_RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        cond0 = 'Stage1'
        cond1 = 'Stage4'

        meth_file = f'{data_dir}CPTAC_data_cpg.csv'
        rna_file = f'{data_dir}CPTAC_data_rna.csv'
        protein_file = f'{data_dir}CPTAC_data_protein.csv'

        meth_sample_file = f'{data_dir}CPTAC_sample_cpg_{cond0}-{cond1}.csv'
        rna_sample_file = f'{data_dir}CPTAC_sample_rna_{cond0}-{cond1}.csv'
        protein_sample_file = f'{data_dir}CPTAC_sample_protein_{cond0}-{cond1}.csv'
        norm_type = 'rows'
        run_type = 'lin'
        rep = '1'
        num_nodes = 5
        mmd_weight = 0.25
        sv = RCMVAE(rcm_file,
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=output_dir,
                    stat_condition_column='stat_condition_label',
                    sub_condition_id_column='stat_condition_id',
                    run_name=f'{num_nodes}-{rep}-{run_type}-{mmd_weight}-{norm_type}',
                    normalise='rows')
        #sv.train_vae(config=config)
        sv.run_vae_stats(include_missing=True)
        # Now reload the
        # meth_df = pd.read_csv(meth_sample_file)
        # rna_df = pd.read_csv(rna_sample_file)
        # s1_cases = list(set(list(meth_df[meth_df['stat_condition_label'] == 'Stage1']['case_id'].values) + list(rna_df[rna_df['stat_condition_label'] == 'Stage1']['case_id'].values)))
        # s4_cases = list(set(list(meth_df[meth_df['stat_condition_label'] == 'Stage4']['case_id'].values) + list(rna_df[rna_df['stat_condition_label'] == 'Stage4']['case_id'].values)))
        # rcm_all_df = pd.read_csv(rcm_file)
        # rcm_labels = ["MDS", "MDS_TMDE", "MDE", "MDE_TMDS", "TMDE", "TMDS", "TPDE", "TPDE_TMDS", "TPDS", "TPDS_TMDE"]
        #
        # for r in rcm_labels:
        #     rcm_df = rcm_all_df[rcm_all_df['Regulation_Grouping_2'] == r]
        #     s1_train_df = rcm_df[['ensembl_gene_id', 'hgnc_symbol', 'entrezgene_id', 'Regulation_Grouping_2']].copy()
        #     s1_dec_df = rcm_df[['ensembl_gene_id', 'hgnc_symbol', 'entrezgene_id', 'Regulation_Grouping_2']].copy()
        #     s1_enc_df = rcm_df[['ensembl_gene_id', 'hgnc_symbol', 'entrezgene_id', 'Regulation_Grouping_2']].copy()
        #
        #     for case in s1_cases:
        #         try:
        #             enc_df, dec_df, train_df = sv.get_decoding_for_cluster(r, [case])
        #             for column in dec_df:
        #                 if column != 'id':
        #                     s1_dec_df[f'{case}_{column}'] = dec_df[column].values
        #             for column in train_df:
        #                 if column != 'id':
        #                     s1_train_df[f'{case}_{column}'] = train_df[column].values
        #             for column in enc_df:
        #                 if column != 'id':
        #                     s1_enc_df[f'{case}_{column}'] = enc_df[column].values
        #             print("Done :) case")
        #
        #         except:
        #             print(case)
        #     s1_train_df.to_csv(f'{f_output_dir}{r}_CPTAC_train_Stage1.csv')
        #     s1_dec_df.to_csv(f'{f_output_dir}{r}_CPTAC_decoding_Stage1.csv')
        #     s1_enc_df.to_csv(f'{f_output_dir}{r}_CPTAC_encoding_Stage1.csv')
        #
        #     s4_train_df = rcm_df[['ensembl_gene_id', 'hgnc_symbol', 'entrezgene_id', 'Regulation_Grouping_2']].copy()
        #     s4___dec_df = rcm_df[['ensembl_gene_id', 'hgnc_symbol', 'entrezgene_id', 'Regulation_Grouping_2']].copy()
        #     s4_enc_df = rcm_df[['ensembl_gene_id', 'hgnc_symbol', 'entrezgene_id', 'Regulation_Grouping_2']].copy()
        #
        #     for case in s4_cases:
        #         try:
        #             enc_df, dec_df, train_df = sv.get_decoding_for_cluster(r, [case])
        #             for column in dec_df:
        #                 if column != 'id':
        #                     s4___dec_df[f'{case}_{column}'] = list(dec_df[column].values.copy())
        #
        #             for column in train_df:
        #                 if column != 'id':
        #                     s4_train_df[f'{case}_{column}'] = train_df[column].values
        #
        #             for column in enc_df:
        #                 if column != 'id':
        #                     s4_enc_df[f'{case}_{column}'] = enc_df[column].values
        #             print("Done :) case")
        #             print("Done :) case")
        #         except:
        #             print(case)
        #     s4_train_df.to_csv(f'{f_output_dir}{r}_CPTAC_train_Stage4.csv')
        #     s4___dec_df.to_csv(f'{f_output_dir}{r}_CPTAC_decoding_Stage4.csv')
        #     s4_enc_df.to_csv(f'{f_output_dir}{r}_CPTAC_encoding_Stage4.csv')

        # # get the gene ids that we're interested in visualising
        # r_df = pd.read_csv(f'{base_dir}output_data/data/MDS-5-1-lin-0.25-rows-Stage1vsStage4.csv')
        #
        # # Get the highest 5 pvalues as a control
        # worst_gene_ids = r_df.nlargest(5, 'Integrated pval (S4-S1)')['ensembl_gene_id'].values
        #
        # # Get the nlargest and nsmallest values from the integrated value
        # r_df = r_df[r_df['Integrated padj (S4-S1)j'] <= 0.05]
        # # Get the top and bottom
        # r_df = r_df[abs(r_df['Integrated diff (S4-S1)']) > 0.1]
        #
        # top_gene_ids = r_df[r_df['Integrated diff (S4-S1)'] > 0]['ensembl_gene_id'].values
        # bot_gene_ids = r_df[r_df['Integrated diff (S4-S1)'] < 0]['ensembl_gene_id'].values
        #
        # # Now we want to plot the top and bottom genes
        #
        # # Get the genes with the largest change between S4 and S1
        # s1_s4_diff = s1_enc_df['VAE'].values - s4_enc_df['VAE'].values
        # plt.hist(s1_s4_diff, bins=20)
        # plt.show()
        # # Plot genes from both
        # # Get top genes by encoding
        # columns = sv.feature_columns
        # # get the genes that chanegd the mosy between
        # cutoff = 1.0
        # hm = Heatmap(s1_train_df[s1_s4_diff > cutoff], row_index='id', chart_columns=columns, cluster_cols=False, cluster_rows=False)
        # hm.axis_line_width = 1.0
        # hm.plot()
        # plt.show()
        #
        # hm = Heatmap(s1_dec_df[s1_s4_diff > cutoff], row_index='id', chart_columns=columns, cluster_cols=False, cluster_rows=False)
        # hm.axis_line_width = 1.0
        # hm.plot()
        # plt.show()
        #
        #hm = Heatmap(s4_train_df[s1_s4_diff > cutoff], row_index='id', chart_columns=columns, cluster_cols=False, cluster_rows=False)
        # hm.axis_line_width = 1.0
        # hm.plot()
        # plt.show()
        #
        # hm = Heatmap(s4_dec_df[s1_s4_diff > cutoff], row_index='id', chart_columns=columns, cluster_cols=False, cluster_rows=False)
        # hm.axis_line_width = 1.0
        # hm.plot()
        # plt.show()

    def test_diff_input_cols(self):
        # scale_data
        data_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/VAE/input_data/'

        output_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/VAE/output_data/stats_files/'
        rcm_file = f'{data_dir}CPTAC_RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        cond0 = 'Stage1'
        cond_runs = ['Stage4'] #, 'Stage3', 'Stage2']

        for cond1 in cond_runs:
            meth_file = f'{data_dir}CPTAC_data_cpg.csv'
            rna_file = f'{data_dir}CPTAC_data_rna.csv'
            protein_file = f'{data_dir}CPTAC_data_protein.csv'

            meth_sample_file = f'{data_dir}CPTAC_sample_cpg_{cond0}-{cond1}.csv'
            rna_sample_file = f'{data_dir}CPTAC_sample_rna_{cond0}-{cond1}.csv'
            protein_sample_file = f'{data_dir}CPTAC_sample_protein_{cond0}-{cond1}.csv'

            epochs = 200  # To make it quicker to train
            batch_size = 16
            num_nodes = 5
            mmd_weight = 0.25
            loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': mmd_weight}
            config = {"loss": loss,
                      "encoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                      "decoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                      "latent": {"num_nodes": 1},
                      "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                      "epochs": epochs,
                      "batch_size": batch_size,
                      "scale_data": False
            }
            norm_type = 'rows'
            sv = RCMVAE(rcm_file,
                        meth_file, meth_sample_file, rna_file, rna_sample_file,
                        protein_file, protein_sample_file, output_folder=output_dir,
                        stat_condition_column='stat_condition_label',
                        sub_condition_id_column='stat_condition_id',
                        run_name=f'{num_nodes}-{1}-{"lin"}-{mmd_weight}-{norm_type}', normalise=norm_type)
            #sv.train_vae(config=config)
            sv.run_vae_stats(f'{cond0}vs{cond1}', include_missing=True)

    def test_gender(self):
        data_dir = 'data/'

        base_dir = '/Users/ariane/Documents/code/sircle_meth/data/S050_CCRCC_Clark_Cell2019/sircle/all_patients/all_cpgs/VAE/alt_comparison/'
        output_dir = f'{base_dir}data/'
        data_dir = f'{base_dir}input/'
        rcm_file = f'{data_dir}CPTAC_RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        cond0 = 'gender'

        meth_file = f'{data_dir}CPTAC_data_cpg.csv'
        rna_file = f'{data_dir}CPTAC_data_rna.csv'
        protein_file = f'{data_dir}CPTAC_data_protein.csv'

        meth_sample_file = f'{data_dir}CPTAC_sample_cpg_Stage1-Stage4.csv'
        rna_sample_file = f'{data_dir}CPTAC_sample_rna_Stage1-Stage4.csv'
        protein_sample_file = f'{data_dir}CPTAC_sample_protein_Stage1-Stage4.csv'
        sv = RCMVAE(rcm_file,
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file, output_folder=output_dir,
                    stat_condition_column='stat_condition_label',
                    sub_condition_id_column='stat_condition_id',
                    run_name=cond0,
                    normalise='rows')
        epochs = 200  # To make it quicker to train
        batch_size = 16
        num_nodes = 5
        mmd_weight = 0.25
        loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': mmd_weight}
        config = {"loss": loss,
                  "encoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "decoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "latent": {"num_nodes": 1},
                  "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                  "epochs": epochs,
                  "batch_size": batch_size,
                  "scale_data": False
                  }
        #sv.train_vae(config=config)
        sv.run_vae_stats(include_missing=True, label='Stage1-Stage4')

    def test_new_stats(self):
        base_dir = 'data/cptac/'
        output_dir = f'{base_dir}'
        data_dir = f'{base_dir}'
        rcm_file = f'{data_dir}RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        label = 'FINAL'

        meth_file = f'{data_dir}CPTAC_cpg.csv'
        rna_file = f'{data_dir}CPTAC_rna.csv'
        protein_file = f'{data_dir}CPTAC_protein.csv'

        meth_sample_file = f'{data_dir}meth_sample_df.csv'
        rna_sample_file = f'{data_dir}rna_sample_df.csv'
        protein_sample_file = f'{data_dir}prot_sample_df.csv'
        sv = RCMStats(rcm_file, f'{data_dir}clinical_CPTAC_TCGA.csv',
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file,
                      output_folder=output_dir, column_id='FullLabel',
                      condition_column='CondId',
                       patient_id_column='SafeCases',
                       run_name=label,
                        normalise='rows', missing_method='clinical', iid=True)
        epochs = 2  # To make it quicker to train
        batch_size = 16
        num_nodes = 5
        mmd_weight = 0.25
        loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': mmd_weight}
        config = {"loss": loss,
                  "encoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "decoding": {"layers": [{"num_nodes": num_nodes, "activation_fn": "relu"}]},
                  "latent": {"num_nodes": 1},
                  "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                  "epochs": epochs,
                  "batch_size": batch_size,
                  "scale_data": False
                  }
        training_cases = ['C3L.00004', 'C3L.00010', 'C3L.00011', 'C3L.00026', 'C3L.00079', 'C3L.00088', 'C3L.00096',
                          'C3L.00097', 'C3L.00103', 'C3L.00360', 'C3L.00369', 'C3L.00416', 'C3L.00418', 'C3L.00447',
                          'C3L.00581', 'C3L.00606', 'C3L.00814', 'C3L.00902', 'C3L.00907', 'C3L.00908', 'C3L.00910', 'C3L.00917', 'C3L.01286', 'C3L.01287', 'C3L.01313', 'C3L.01603', 'C3L.01607', 'C3L.01836', 'C3N.00148', 'C3N.00149', 'C3N.00150', 'C3N.00177', 'C3N.00194', 'C3N.00244', 'C3N.00310', 'C3N.00314', 'C3N.00390', 'C3N.00494', 'C3N.00495', 'C3N.00573', 'C3N.00577', 'C3N.00646', 'C3N.00733', 'C3N.00831', 'C3N.00834', 'C3N.00852', 'C3N.01176', 'C3N.01178', 'C3N.01179', 'C3N.01200', 'C3N.01214', 'C3N.01220', 'C3N.01261', 'C3N.01361', 'C3N.01522', 'C3N.01646', 'C3N.01649', 'C3N.01651', 'C3N.01808']
        sv.train_vae(cases=training_cases, config=config)
        sv.save()
        #sv.load_saved_vaes()
        #sv.load_saved_inputs(f'{output_dir}input_df_{label}.csv')
        #sv.load_saved_encodings(f'{output_dir}encoded_df_{label}.csv')
        sv.run_vae_stats(cond_label='gender', cond0='Male', cond1='Female')
        #sv.run_vae_stats(cond_label='AgeGrouped', cond0='young', cond1='old')
        #sv.run_vae_stats(cond_label='TumorStage', cond0='Stage I', cond1='Stage IV')
        dec = sv.get_decoding('MDS')
        print(dec)