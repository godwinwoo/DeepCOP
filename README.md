# DeepCOP - Deep gene COmpound Profiler

This is the codebase that was used to obtain the results in the corresponding paper: https://www.ncbi.nlm.nih.gov/pubmed/31504186.

Use steps 1-4 to train and validate an MLP model to predict the gene expression of a gene given a molecule on a particular cell line of the LINCS L1000 dataset. Step 5 is used to evaluate the trained models against actual RNA-Seq values.

## Data Preparation
1. Download and uncompress the level 5 gctx data files and experiment metadata from GEO
   * Phase 1: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742
     * GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz <br>
     * GSE92742_Broad_LINCS_sig_info.txt.gz <br>
   * Phase 2: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138 <br>
     * GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz <br>
     * GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz <br>

2. The rar files in the Data folder are large so they had to be compressed to github. 
Uncompress these files in the same folder. 

3. Use [get_xy.py](get_xy.py) or [get_xy_phase2.py](get_xy_phase2.py) to collect the training data and labels.
    * You will need to set the LINCS_data_path to the folder where you uncompressed the gctx data files.

## Train and Validate MLP Models
4. Use [internal_validation.py](internal_validation.py) to do 10 fold cross validation on the training data.
    * This step will save the trained models to SavedModels folder as well as the cutoff values. 
    The cutoff values are saved when all 10 folds are evaluated. The cutoffs are used in step 5.  

## Evaluate Trained Model Predictions against Actual RNA-Seq Values
5. Use [external_validation.py](external_validation.py) to evaluate predictions from step 3 trained models on external RNA-Seq values.
    * When evaluating the predictions, cutoff values from step 4 will be used.

## Extra
  * We used [gene_descriptors.r](Extra/gene_descriptors.r) to compute gene descriptors. This was done using R. <br>
  * [rdkit_fingerprint.py](Extra/rdkit_fingerprint.py) was used to compute morgan fingerprints. <br>
  * [application_domain.py](application_domain.py) was used to plot the Jaccard similarities
