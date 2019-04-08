from Helpers.data_loader import get_feature_dict, load_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score
import seaborn as sns

def remove_non_lncap(dict):
    lncap_drugs = load_csv('data/LNCAPdrugs.csv')
    new_dict = {}
    for key in dict:
        if [key] in lncap_drugs:
            new_dict[key] = dict[key]
    return new_dict


# remove duplicate drugs
def remove_dups(dict):
    unique_dict = {}
    check_dups = {}
    for drug_id in dict:
        value = dict[drug_id]
        val_str = ''.join(value)
        if val_str in check_dups:
            continue
        check_dups[val_str] = 1

        drug_features = np.asarray(value, dtype='float16')
        unique_dict[drug_id] = drug_features
    return unique_dict


def remove_dup_np(arr):
    checkdup = []
    unique_arr = []
    for row in arr:
        val_str = ''.join(str(int(x)) for x in row)
        if val_str in checkdup:
            continue
        checkdup.append(val_str)
        unique_arr.append(row)
    return np.asarray(unique_arr)


def get_array(dict):
    list = []
    for key in dict:
        value = dict[key]
        list.append(value)

    return np.array(list)


def remove_corr_features(all_features):
    uneeded_cols = load_csv('Data/LNCAPcorr_cols.csv')
    uneeded_cols_int = []
    for col in uneeded_cols:
        uneeded_cols_int.append(int(col[0]))

    print(len(uneeded_cols))
    n_cols = all_features.shape[1]
    cols = range(0, n_cols)
    cols = [x for x in cols if x not in uneeded_cols_int]
    return all_features[:, cols]


def get_uncorr_drug_features(include_rnaseq=True):
    # first get the compounds that are from the LINCS dataset
    drug_features_dict = get_feature_dict('Data/LDS1484_compounds_morgan_2048_nk.csv')  # , use_int=True)
    drug_features_dict = remove_non_lncap(drug_features_dict)

    # add the compounds that were in RNAseq
    if include_rnaseq:
        rnaseq_drugs = get_feature_dict('Data/rnaseq_morgan_2048_nk.csv')
        for rnaseq_drug in rnaseq_drugs:
            drug_features_dict[rnaseq_drug] = rnaseq_drugs[rnaseq_drug]

    unique_drug_features_dict = remove_dups(drug_features_dict)
    drug_features = get_array(unique_drug_features_dict)
    no_cor_np = remove_corr_features(drug_features)
    return remove_dup_np(no_cor_np)


def get_jaccard_score_of_rnaseq_drug(drug_id, lincs_drugs):
    rnaseq_drugs = get_feature_dict('Data/rnaseq_morgan_2048_nk.csv')

    rnaseq_drug = rnaseq_drugs[drug_id]
    rnaseq_drug = np.reshape(np.array(rnaseq_drug, np.float16), (1, -1))
    rnaseq_drug = remove_corr_features(rnaseq_drug)
    scores = []
    for lincs_drug in lincs_drugs:
        score = jaccard_similarity_score(lincs_drug, rnaseq_drug[0])
        scores.append(score)
    return np.mean(scores)


def get_jaccard_scores():
    lincs_drug_features = get_uncorr_drug_features(False)
    num_drugs = len(lincs_drug_features)

    # get the scores from each other
    scores = []
    for i in range(0, num_drugs):
        for j in range(i+1, num_drugs):
            source_drug = lincs_drug_features[i]
            target_drug = lincs_drug_features[j]
            score = jaccard_similarity_score(source_drug, target_drug)
            scores.append(score)

    plot = sns.distplot(scores, bins=100, axlabel="Jaccard Similarity Coefficient", norm_hist=False)
    plot.set_title("Histogram of Jaccard Similarity Scores Between Trained Compounds")
    plot.set(ylabel="Count")
    plot.ticklabel_format(style='plain')  # , axis='both', scilimits=(0, 0))

    score_enza = get_jaccard_score_of_rnaseq_drug("Enzalutamide", lincs_drug_features)
    score_17005 = get_jaccard_score_of_rnaseq_drug("VPC17005", lincs_drug_features)
    score_14449 = get_jaccard_score_of_rnaseq_drug("VPC14449", lincs_drug_features)
    plt.plot([score_enza, score_enza], [0, 25], color="yellow", label="Enzalutamide")
    plt.plot([score_17005, score_17005], [0, 25], color="red", label="VPC-17005")
    plt.plot([score_14449, score_14449], [0, 25], color="blue", label="VPC-14449")

    plot.legend(loc="upper right")

    fig = plot.get_figure()
    fig.show()


get_jaccard_scores()
