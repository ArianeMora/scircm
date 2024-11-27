# sci-RegulatoryClusteringModel
[![PyPI](https://img.shields.io/pypi/v/scircm)](https://pypi.org/project/scircm/)

## Versions: 
1. [R version](https://github.com/ArianeMora/SiRCleR)!
2. [Python without VAE](https://github.com/ArianeMora/SiRClePy)!

## Install
Optionally create a new conda env.
```
conda create --name scircm python=3.8
conda activate scircm
pip install scircm
```

## Run
See the [examples](https://github.com/ArianeMora/scircm/tree/main/examples)! folder for a proper tutorial with data included that you can run!

*If you have any troubles running the tutorial on a windows machine, please let us know and we can help to sort out any issues. 

## Issues
Please let us know if you have any issues (ariane.n.mora@gmail.com) or via the [issues](https://github.com/ArianeMora/scircm/issues)! tab (preferred).

#### Quick version
```
from scircm import SciRCM
# FORMAT must be csv :) 
prot_file = f'path to the output from protein differential abundence file'
rna_file = f'path to the output from differential expression analysis file'
meth_file = f'path to the output from methylation DCpG analysis file'

# Note we assume your methylation CpGs map to a single gene, if they don't see the section below.
# logFC_rna = column name in your RNA file that has your RNA logFC (same for the protein and CpG)
# padj_rna = column name in your RNA file that has your padj value (same for protein and CpG)
# NOTE: these need to be unique from one another since we merge the datasets, if they aren't, you need
# to update your csv files.
# Lastly: ensembl_gene_id this is the gene ID column, All must use the same identifier, and this must be
# labelled the same in each file, if it isn't, update your column names before running.

rcm = SciRCM(meth_file, rna_file, prot_file, 
             "logFC_rna", "padj_rna", "CpG_Beta_diff", "padj_meth", "logFC_protein", "padj_protein",
             "ensembl_gene_id", sep=',',
             rna_padj_cutoff=0.05, 
             prot_padj_cutoff=0.05, 
             meth_padj_cutoff=0.05,
             rna_logfc_cutoff=1.0, 
             prot_logfc_cutoff=0.5, 
             meth_diff_cutoff=0.1, 
             output_dir='',
             non_coding_genes=['None'],
             output_filename='RCM_Output.csv',
             bg_type = '(P&M)|(P&R)|(M&R)'
         )
rcm.run()
df = rcm.get_df()
# That DF now has your rcm clustering results, how easy was that :D
```

#### Making your CpGs map to a single gene version
```
from scircm import filter_methylation_data_by_genes
meth_df = pd.read_csv(f'path to the output from methylation DCpG analysis file')
# Note: you need to pass it: 
# 1) the gene ID column, here it is 'ensembl_gene_id'
# 2) the padj column: here it is 'padj_meth'
# 3) the logFC or test statistic column: here it is 'CpG_Beta_diff'
filtered_meth_df = filter_methylation_data_by_genes(meth_df, 'ensembl_gene_id', 'padj_meth', 'CpG_Beta_diff')
```

## Manuscript

If you want to read more about how SiRCle works, please check out our paper: https://www.biorxiv.org/content/10.1101/2022.07.02.498058v1 

Website is available at: https://arianemora-sircle-web-app-ndu996.streamlit.app it is under active development and currently in Beta mode - let us know if you have any issues :) 
Note the website is only for the first bit of the regulatory clustering.


Now you can run the first version :)

## Quick guide on the regulation grouping levels

We have four levels of regulation grouping, each encoding a different level of clustering, the excel spreadsheet in examples 
has all these written explicitly. We include four levels so that depending on the experiment types and specific data 
used (e.g. type of proteomics) the user can choose the right granularity.

1. Regulation Grouping All: All 54 levels where the labels are just the level of DNA methylation, RNA and protein.
2. Regulation Grouping Change: 54 groups consolidated to 10 such that we prioritise "changes" in the system. For example, an "UP" on RNA followed by "Not significant" or not meeting the threshold will consider that a "double regulation", namely, an "increase" on the RNA level followed by a suppresion on the translational level (because it didn't meet the threshold).
3. Regulation Grouping Detection: 54 groups consolidated to 10 such that we prioritise the central dogma and "stop" the labelling at the last significant change. For example, the above example ("UP" on RNA followed by "NS" on protein) would just be Transcription driven increase (TPDE).
4. Regulation Grouping Protein: 54 groups consolidated to 6 clusters, such that we only consider one layer of regulation (no double groups) and that if there are two opposing changes (for example "UP" on the transcriptional layer followed by "DOWN" on the protein, it would be labelled via the "DOWN" on the protein layer (i.e. TMDS)


## Quick guide on the background filter
To compensate for detection thresholds (e.g. the protein layer having much less than RNA or Methylation) we include a background filter that "Nones" labels that don't meet the threshold.

The background options included are:
1. 'P&R': Protein and RNA exist for that gene, this is the recommended level of background as it avoids confusion with detection limits.
2. '(P&M)|(P&R)|(M&R)': Protein and DNA methylation, or, Protein and RNA, or Methylation and RNA exist in the datasets for that gene. This is recommended for if you are interested in layers that may not have protein, for example including non-coding genes or have poor detection on the protein level.
3. '*': No filter
4. 'P&M&R': The strictest one, the gene must have Protein and RNA and DNA methylation (this one is only recomended if you really want the impact of DNA methylation to be the focus)
5. 'P|M|R': Akin to "*" unless an added statistical filter is added (see footnote)
6. '(P&M)|(P&R)':  Protein and Methylation or protein and RNA.
7. 'P|(M&R)': Protein or RNA and Methylation exist.
8. 'P|R': Protein or RNA exists in the dataset.

* Note, when running the function `run_rcm(self, methylation_background=1.0, rna_background=1.0, protein_background=1.0)` you can pass different thresholds to the background filter, the current filter is just that the p.adj exists (i.e. is not None), however, one could pass 0.05 if one wanted the background to only consider genes with a significant change.

## Issues

### Note on Mac M1
Unfortunately, the Mac M1 has some issues with tensorflow, so we have not been able to get our package to work on a M1 
mac yet. If you get it working please let us know your solution as others may benefit! We have a partial M1 version
(that covers the regulatory clustering) but doesn't enable the group statistics, if you are only interested in part 1 
of SiRCle please go here: https://github.com/ArianeMora/SiRCle_M1

### Note on Windows
We have tested our code on Windows (10) and Mac (pro) computers, I'm not sure how it would go on a Windows 7 machine so 
if you have issues post an issue.

### Note on libraries/dependenices
If you find that things don't install please let us know! We've done our best to make it reproducible but different 
environments may mess things up so we're happy to help you debug, just post an issue on the github.

Note we expect python 3.8 so if things don't work first time, check you're running python 3.8 and then try again :) 

#### Things to note

- As at 27/11 we updated the names of the clusters and included an extra level of grouping that explicitly takes into account whether the protein was detected or not


## Regulatory clustering model 

The general table of how we define regulatory clusters.

Please post questions and issues related to sci-rcm on the `Issues <https://github.com/ArianeMora/scircm/issues>`_  section of the GitHub repository.


## Cite
If you use this please cite our manuscript: https://www.biorxiv.org/content/10.1101/2022.07.02.498058v1 