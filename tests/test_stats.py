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
from scircm import RCMStats


class TestClass(unittest.TestCase):

    @classmethod
    def setup_class(self):
        local = True
        # Create a base object since it will be the same for all the tests
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data_dir = os.path.join(THIS_DIR, 'data')
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

    def test_run(self):
        base_dir = '/Users/ariane/Documents/code/roundround/data/sircle_pipeline/sircle_input/stats/'
        output_dir = f'{base_dir}'
        data_dir = f'{base_dir}'
        rcm_file = f'{data_dir}RCM_Renal_cell_carcinoma__NOS.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        label = 'FINAL'

        meth_file = f'{data_dir}CPTAC_cpg_Renal_cell_carcinoma__NOS.csv'
        rna_file = f'{data_dir}CPTAC_rna_Renal_cell_carcinoma__NOS.csv'
        protein_file = f'{data_dir}CPTAC_protein_Renal_cell_carcinoma__NOS.csv'

        meth_sample_file = f'{data_dir}samples_CpG_Renal_cell_carcinoma__NOS.csv'
        rna_sample_file = f'{data_dir}samples_RNA_Renal_cell_carcinoma__NOS.csv'
        protein_sample_file = f'{data_dir}samples_protein_Renal_cell_carcinoma__NOS.csv'
        sv = RCMStats(rcm_file, f'{data_dir}patient_info_Renal_cell_carcinoma__NOS.csv',
                    meth_file, meth_sample_file, rna_file, rna_sample_file,
                    protein_file, protein_sample_file,
                      output_folder=output_dir, column_id='FullLabel',
                      condition_column='CondID',
                       patient_id_column='SafeCases',
                       run_name=label,
                      regulatory_label='RG2_Changes_filtered',
                        normalise='rows', missing_method='mean', iid=False)
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
        training_cases = ['C3L.00004', 'C3L.00004', 'C3L.00010', 'C3L.00010', 'C3L.00026',
       'C3L.00026', 'C3L.00079', 'C3L.00079', 'C3L.00103', 'C3L.00103',
       'C3L.00360', 'C3L.00360', 'C3L.00369', 'C3L.00369', 'C3L.00416',
       'C3L.00416', 'C3L.00581', 'C3L.00581', 'C3L.00583', 'C3L.00583',
       'C3L.00606', 'C3L.00606', 'C3L.00607', 'C3L.00607', 'C3L.00814',
       'C3L.00814', 'C3L.00902', 'C3L.00902', 'C3L.00907', 'C3L.00907',
       'C3L.00908', 'C3L.00908', 'C3L.00910', 'C3L.00910', 'C3L.00917',
       'C3L.00917', 'C3L.01302', 'C3L.01302', 'C3L.01313', 'C3L.01313',
       'C3L.01607', 'C3L.01607', 'C3L.01836', 'C3L.01836', 'C3N.00150',
       'C3N.00150', 'C3N.00168', 'C3N.00168', 'C3N.00177', 'C3N.00177',
       'C3N.00310', 'C3N.00310', 'C3N.00314', 'C3N.00314', 'C3N.00390',
       'C3N.00390', 'C3N.00495', 'C3N.00495', 'C3N.00646', 'C3N.00646',
       'C3N.00733', 'C3N.00733', 'C3N.00831', 'C3N.00831', 'C3N.00852',
       'C3N.00852', 'C3N.01176', 'C3N.01176', 'C3N.01178', 'C3N.01178',
       'C3N.01214', 'C3N.01214', 'C3N.01524', 'C3N.01524', 'C3N.01646',
       'C3N.01646']
        sv.train_vae(cases=training_cases, config=config)
        sv.save()
        sv.load_saved_vaes()
        sv.load_saved_inputs(f'{output_dir}vae_input_df_{label}.csv')
        sv.load_saved_encodings(f'{output_dir}encoded_df_{label}.csv')
        sv.run_vae_stats(cond_label='gender', cond0='Male', cond1='Female')
        sv.run_vae_stats(cond_label='AgeGrouped', cond0='young', cond1='old')
        sv.run_vae_stats(cond_label='TumorStage', cond0='Stage I', cond1='Stage IV')
        dec = sv.get_decoding('MDS')
        print(dec)


    def test_new_stats(self):
        base_dir = '../examples/data/'
        output_dir = f'{base_dir}'
        data_dir = f'{base_dir}'
        rcm_file = f'{data_dir}RCM.csv'
        # Now we want to check stats where we use the data that accomnaied those inputs
        label = 'FINAL'

        meth_file = f'{data_dir}CPTAC_cpg_S4.csv'
        rna_file = f'{data_dir}CPTAC_rna_S4.csv'
        protein_file = f'{data_dir}CPTAC_protein_S4.csv'

        meth_sample_file = f'{data_dir}cpg_sample_data_Stage IV_sircle.csv'
        rna_sample_file = f'{data_dir}rna_sample_data_Stage IV_sircle.csv'
        protein_sample_file = f'{data_dir}prot_sample_data_Stage IV_sircle.csv'
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
                          'C3L.00581', 'C3L.00606', 'C3L.00814', 'C3L.00902', 'C3L.00907', 'C3L.00908', 'C3L.00910',
                          'C3L.00917', 'C3L.01286', 'C3L.01287', 'C3L.01313', 'C3L.01603', 'C3L.01607', 'C3L.01836',
                          'C3N.00148', 'C3N.00149', 'C3N.00150', 'C3N.00177', 'C3N.00194', 'C3N.00244', 'C3N.00310',
                          'C3N.00314', 'C3N.00390', 'C3N.00494', 'C3N.00495', 'C3N.00573', 'C3N.00577', 'C3N.00646',
                          'C3N.00733', 'C3N.00831', 'C3N.00834', 'C3N.00852', 'C3N.01176', 'C3N.01178', 'C3N.01179',
                          'C3N.01200', 'C3N.01214', 'C3N.01220', 'C3N.01261', 'C3N.01361', 'C3N.01522', 'C3N.01646',
                          'C3N.01649', 'C3N.01651', 'C3N.01808']
        sv.train_vae(cases=training_cases, config=config)
        sv.save()
        sv.load_saved_vaes()
        sv.load_saved_inputs(f'{output_dir}vae_input_df_{label}.csv')
        sv.load_saved_encodings(f'{output_dir}encoded_df_{label}.csv')
        sv.run_vae_stats(cond_label='gender', cond0='Male', cond1='Female')
        sv.run_vae_stats(cond_label='AgeGrouped', cond0='young', cond1='old')
        sv.run_vae_stats(cond_label='TumorStage', cond0='Stage I', cond1='Stage IV')
        dec = sv.get_decoding('MDS')
        print(dec)

    def test_stats(self):
        data_dir = '../examples/data/'
        missing_method = 'mean'
        label = missing_method
        sv = RCMStats(rcm_file=f'{data_dir}RCM.csv',
                      patient_sample_file=f'{data_dir}clinical_CPTAC_TCGA.csv',
                      # Clinical file for all patients in the study
                      meth_file=f'{data_dir}CPTAC_cpg.csv',
                      meth_sample_file=meth_sample_file,
                      rna_file=f'{data_dir}CPTAC_rna.csv',
                      rna_sample_file=rna_sample_file,
                      protein_file=f'{data_dir}CPTAC_protein.csv',
                      protein_sample_file=prot_sample_file,
                      output_folder=data_dir,
                      regulatory_label='RG2_Changes_filtered',
                      column_id='FullLabel',
                      condition_column='CondId',
                      patient_id_column='SafeCases',  # This is the column that is in each of the sample DFs
                      run_name=label,
                      normalise='rows',
                      verbose=True,
                      missing_method=missing_method)

        # Check out the patient info
        # Get the patient info that has been compiled from the provided sample files
        patient_info = sv.patient_clinical_df
        # Select the cases with 5 samples
        matching_patient_info = patient_info[patient_info['Sample counts'] == 5]
        matching_cases = matching_patient_info['SafeCases'].values
        print("total number of patients: ", len(patient_info), " vs number with matching data: ",
              len(matching_patient_info))
        train = True
        if train:
            epochs = 100
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
            training_cases = matching_cases  # Use matching cases for training
            sv.train_vae(cases=matching_cases, config=config)
            sv.save()  # Save the information we have generated.
        else:
            sv.load_saved_vaes()
            sv.load_saved_encodings(f'{sv.output_folder}encoded_df_{label}.csv')
            sv.load_saved_inputs(f'{sv.output_folder}vae_input_df_{label}.csv')
            sv.load_saved_raws(f'{sv.output_folder}raw_input_df_{label}.csv')