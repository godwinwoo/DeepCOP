import datetime
import json
import time
import numpy as np
from Helpers.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar

start_time = time.time()
gene_count_data_limit = 978
target_cell_names = ['LNCAP']
test_blind = True
save_xy_path = "TrainData/"
LINCS_data_path = "/data/datasets/gwoo/L1000/LDS-1484/Data/"  # set this path to your LINCS gctx file
data_path = "Data/"
gap_factors = [0.0]
significance_levels = [5]
dosage = 10


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


def get_gene_id_dict():
    lm_genes = json.load(open('Data/landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict


def get_class_vote(pert_list, bottom_threshold, top_threshold):
    votes = [0, 0, 0, 0]
    # list of perts
    for pert in pert_list:
        if pert > top_threshold:
            votes[3] += 1  # upregulation
        elif pert < bottom_threshold:
            votes[1] += 1  # downregulation
        else:
            votes[2] += 1  # not regulated
    highest_vote_class = np.argmax(votes)

    is_tie = False  # check if there's a tie (another class with the same number of votes)
    for i in range(0, len(votes)):
        if i == highest_vote_class:
            continue
        if votes[i] == votes[highest_vote_class]:
            is_tie = True
            break
    if is_tie:
        return 0
    else:
        return highest_vote_class


def get_their_id(good_id):
    return 'b\'' + good_id + '\''


def get_our_id(bad_id):
    return bad_id[2:-1]


# get the dictionaries
print(datetime.datetime.now(), "Loading drug and gene features")
drug_features_dict = get_feature_dict('Data/phase2_compounds_morgan_2048.csv')
gene_features_dict = get_feature_dict('Data/go_fingerprints.csv')
cell_name_to_id_dict = get_feature_dict('Data/Phase2_Cell_Line_Metadata.txt', '\t', 2)
experiments_dose_dict = get_feature_dict(LINCS_data_path + 'GSE70138_Broad_LINCS_sig_info.txt', '\t', 0)
gene_id_dict = get_gene_id_dict()

lm_gene_entrez_ids = []
for gene in gene_id_dict:
    lm_gene_entrez_ids.append(get_their_id(gene))

print("Loading gene expressions from gctx")
level_5_gctoo = load_gene_expression_data(LINCS_data_path + "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx",
                                          lm_gene_entrez_ids)
length = len(level_5_gctoo.col_metadata_df.index)
# length = 10000

for target_cell_name in target_cell_names:
    target_cell_id = cell_name_to_id_dict[target_cell_name][0]

    cell_X = {}  # stores rows used in X
    cell_Y = {}  # stores the highest perturbation for that experiment
    cell_Y_gene_ids = []
    cell_drugs_counts = 0
    repeat_X = {}

    print("Loading experiments")
    # For every experiment
    for i in range(0, length):
        printProgressBar(i, length, prefix='Load experiments progress')
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]

        # parse the time
        start = col_name.rfind("_")
        end = find_nth(col_name, ":", 1)
        exposure_time = col_name[start + 1:end]
        if exposure_time != "24H":  # column counts: 6h 95219, 24h 109287, 48h 58, 144h 1
            continue

        # get drug features
        col_name_key = col_name[2:-1]
        if col_name_key not in experiments_dose_dict:
            continue
        experiment_data = experiments_dose_dict[col_name_key]
        drug_id = experiment_data[0]
        if drug_id not in drug_features_dict:
            continue
        drug_features = drug_features_dict[drug_id]

        # parse the dosage unit and value
        dose_unit = experiment_data[4][-2:]
        if dose_unit != 'um':
            # remove any dosages that are not 'µM'. Want to standardize the dosages.
            # column counts: -666 17071, % 2833, uL 238987, uM 205066, ng 1439, ng / uL 2633, ng / mL 5625
            continue
        dose_amt = float(experiment_data[4][:-2])
        if dose_amt < dosage - 0.1 or dose_amt > dosage + 0.1:  # 10µM +/- 0.1
            continue

        # parse the cell name
        start = find_nth(col_name, "_", 1)
        end = find_nth(col_name, "_", 2)
        cell_name = col_name[start + 1:end]
        if cell_name != target_cell_name:
            continue

        if cell_name not in cell_name_to_id_dict:
            continue
        cell_id = cell_name_to_id_dict[cell_name][0]

        for gene_id in lm_gene_entrez_ids:
            our_gene_id = get_our_id(gene_id)
            gene_symbol = gene_id_dict[our_gene_id]

            if gene_symbol not in gene_features_dict:
                continue

            pert = column[gene_id].astype('float16')
            # repeat key is used to find the largest perturbation for similar experiments and filter out the rest
            repeat_key = drug_id + "_" + our_gene_id

            if repeat_key not in cell_X:
                cell_X[repeat_key] = drug_features + gene_features_dict[gene_symbol]
                cell_Y[repeat_key] = []
                cell_Y_gene_ids.append(our_gene_id)
                cell_drugs_counts += 1
            cell_Y[repeat_key].append(pert)

    elapsed_time = time.time() - start_time
    print(datetime.datetime.now(), "Time to load data:", elapsed_time)

    # at this point all the perturbation values are stored in cell_Y
    # now we want to determine the perturbation classes

    gene_cutoffs_down = {}
    gene_cutoffs_up = {}
    for percentile_down in significance_levels:
        percentile_up = 100 - percentile_down

        # get all the global gene_specific_cutoffs:
        prog_ctr = 0
        for gene_id in lm_gene_entrez_ids:
            prog_ctr += 1
            printProgressBar(prog_ctr, gene_count_data_limit, prefix='Storing percentile cutoffs')
            our_gene_id = get_our_id(gene_id)
            their_gene_id = gene_id

            row = level_5_gctoo.data_df.loc[their_gene_id, :].values
            gene_cutoffs_down[our_gene_id] = np.percentile(row, percentile_down)
            gene_cutoffs_up[our_gene_id] = np.percentile(row, percentile_up)

        # get voting class
        print(datetime.datetime.now(), "Converting dictionary values to numpy array, count", str(len(cell_X)))
        npX = np.asarray(list(cell_X.values()), dtype='float16')
        y_pert_lists = np.asarray(list(cell_Y.values()))
        n_samples = len(y_pert_lists)
        npY_class_down = np.zeros(n_samples, dtype=int)
        npY_class_up = np.zeros(n_samples, dtype=int)
        npY_gene_ids = np.asarray(cell_Y_gene_ids)

        prog_ctr = 0
        combined_locations = []
        combined_test_locations = []
        for gene_id in lm_gene_entrez_ids:  # this section is for gene specific class cutoffs
            prog_ctr += 1
            printProgressBar(prog_ctr, gene_count_data_limit, prefix='Marking positive pertubations')
            our_gene_id = get_our_id(gene_id)
            gene_locations = np.where(npY_gene_ids == our_gene_id)
            class_cut_off_down = gene_cutoffs_down[our_gene_id]
            class_cut_off_up = gene_cutoffs_up[our_gene_id]

            voted_classes = np.zeros(n_samples, dtype=int)
            for pert_i in gene_locations[0]:
                voted_classes[pert_i] = get_class_vote(y_pert_lists[pert_i], class_cut_off_down, class_cut_off_up)

            down_locations = np.where(voted_classes == 1)
            mid_locations = np.where(voted_classes == 2)
            up_locations = np.where(voted_classes == 3)

            npY_class_down[down_locations] = 1  # set all down class locations to be active
            npY_class_up[up_locations] = 1  # set all up class locations to be active

        print('Positive samples down', np.sum(npY_class_down))
        print('Positive samples up',  np.sum(npY_class_up))
        num_drugs = cell_drugs_counts
        print("Sample Size:", n_samples, "Drugs tested:", num_drugs / gene_count_data_limit)

        # save the data to be loaded and used multiple times if needed
        model_file_prefix = save_xy_path + target_cell_name + '_' + str(percentile_down) + 'p'
        save_file = model_file_prefix + "_X"
        print("saved", save_file)
        np.savez(save_file, npX)

        for direction in ["Down", "Up"]:
            model_file_prefix = save_xy_path + target_cell_name + '_' + direction + '_' + str(percentile_down) + 'p'
            save_file = model_file_prefix + "_Y_class"
            print("saved", save_file)
            npY = npY_class_up if direction == "up" else npY_class_down
            np.savez(save_file, npY)
