import numpy as np
from pathlib import Path
from Helpers.data_loader import get_feature_dict, load_csv
from keras.models import model_from_json
from Helpers.utilities import all_stats
import sklearn.metrics as metrics

lincs_to_rnaseq_gene = {
        'PAPD7': 'TENT4A',
        'HDGFRP3': 'HDGFL3',
        'TMEM2': 'CEMIP2',
        'TMEM5': 'RXYLT1',
        'SQRDL': 'SQOR',
        'KIAA0907': 'KHDC4',
        'IKBKAP': 'ELP1',
        'TMEM110': 'STIMATE',
        'NARFL': 'CIAO3',
        'HN1L': 'JPT2'
    }


# load model
def load_model(file_prefix):
    # load json and create model
    json_file = open(file_prefix + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_prefix + '.h5')
    print("Loaded model", file_prefix)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def load_model_from_file_prefix(model_file_prefix):
    model_file = Path(model_file_prefix + ".json")
    if not model_file.is_file():
        print(model_file.name + "File not found")
    return load_model(model_file_prefix)


def get_predictions(up_model_filename, down_model_filename):
    # load the models
    up_model = load_model_from_file_prefix(up_model_filename)
    down_model = load_model_from_file_prefix(down_model_filename)

    # build your input features
    gene_features_dict = get_feature_dict("Data/go_fingerprints.csv")
    drug_features_dict = get_feature_dict("Data/inhouse_morgan_2048.csv")

    data = []
    descriptions = []
    rnaseq_missing_genes = [  # these genes were not in the rnaseq dataset
        'GATA3',
        'RPL39L',
        'IKZF1',
        'CXCL2',
        'HMGA2',
        'TLR4',
        'SPP1',
        'MEF2C',
        'PRKCQ',
        'MMP1',
        'PTGS2',
        'ICAM3',
        'INPP1',
    ]
    for gene in rnaseq_missing_genes:
        gene_features_dict.pop(gene, None)
    for drug in drug_features_dict:
        for gene in gene_features_dict:
            data.append(drug_features_dict[drug] + gene_features_dict[gene])
            descriptions.append(drug + ", " + gene)
    data = np.asarray(data, dtype=np.float16)

    # get predictions
    up_predictions = up_model.predict(data)
    down_predictions = down_model.predict(data)

    return up_predictions, down_predictions, drug_features_dict, gene_features_dict


def get_true_from_padj(drugs, genes, old_to_new_symbol, rnaseq_data, significance_level):
    up_true_float = []
    down_true_float = []
    up_true_int = []
    down_true_int = []

    for drug in drugs:
        for gene in genes:
            if gene in old_to_new_symbol:
                gene = old_to_new_symbol[gene]
            if gene not in rnaseq_data[drug]:
                print('rnaseq missing gene', gene)
                continue
            padj = float(rnaseq_data[drug][gene][1])
            log2change = float(rnaseq_data[drug][gene][0])
            up_value = 0
            down_value = 0
            if log2change >= 0:
                if padj <= significance_level:
                    up_value = 1
                up_true_float.append(-padj)
                down_true_float.append(-1)
                up_true_int.append(up_value)
                down_true_int.append(0)
            else:
                if padj <= significance_level:
                    down_value = 1
                up_true_float.append(-1)
                down_true_float.append(-padj)
                up_true_int.append(0)
                down_true_int.append(down_value)
    return up_true_float, down_true_float, up_true_int, down_true_int


def print_acc(text, Y_train, y_pred_train):
    y_pred = np.argmax(y_pred_train, axis=1)
    y_true = Y_train
    target_names = [0, 1]
    cm = metrics.confusion_matrix(y_true, y_pred, labels=target_names)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accs = cm.diagonal()
    print(text, "Accuracy class 0", accs[0])  # number of actual 0's predicted correctly
    print(text, "Accuracy class 1", accs[1])  # number of actual 1's predicted correctly

    report = metrics.classification_report(y_true, y_pred)
    print("Report", report)


def print_stats(y_true, param, dir, predictions, cutoff=None):
    val_stats = all_stats(np.asarray(y_true, dtype='float32'), predictions[:, 1], cutoff)
    label = dir + "regulation " + str(param)
    print(label)
    print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff | Max F Score')
    print('All stats val:', ['{:6.3f}'.format(val) for val in val_stats])
    print_acc(label, np.asarray(y_true, dtype='float32'), predictions)


def compare_predictions_with_rnaseq(up_model_filename, down_model_filename, upcutoff, downcutoff):
    # get the predictions np array ordered by drugs then genes
    up_predictions, down_predictions, drugs, genes = get_predictions(up_model_filename, down_model_filename)

    # get the rnaseq data into np array
    csv_file = load_csv('Data/DESeq2results.csv')  # this data is found at GSE127816
    rnaseq_data = {}
    for line in csv_file[1:]:
        drug = line[0]
        gene = line[1]
        padj = line[2:5]
        if drug not in rnaseq_data:
            rnaseq_data[drug] = {}
        if gene not in rnaseq_data[drug]:
            rnaseq_data[drug][gene] = padj

    significance_level = 0.05
    print("significance level", significance_level)

    up_true_float, down_true_float, up_true_int, down_true_int = \
        get_true_from_padj(drugs, genes, lincs_to_rnaseq_gene, rnaseq_data, significance_level)

    print_stats(up_true_int, significance_level, "up", up_predictions, upcutoff)
    print_stats(down_true_int, significance_level, "down", down_predictions, downcutoff)


def get_average_of_10_models():
    up_file_prefix = "SavedModels/LNCAP_Up_5p_"
    down_file_prefix = "SavedModels/LNCAP_Down_5p_"
    up_cutoffs = np.load(up_file_prefix + "cutoffs.npz")['arr_0']
    down_cutoffs = np.load(down_file_prefix + "cutoffs.npz")['arr_0']

    for i in range(0, 10):
        up_model_filename = up_file_prefix + str(i+1)
        down_model_filename = down_file_prefix + str(i+1)
        upcutoff = up_cutoffs[i]
        downcutoff = down_cutoffs[i]
        compare_predictions_with_rnaseq(up_model_filename, down_model_filename, upcutoff, downcutoff)


get_average_of_10_models()
