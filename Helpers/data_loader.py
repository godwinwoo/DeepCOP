from cmapPy.pandasGEXpress import parse
import numpy as np
import csv
import Helpers.remove_correlated_features as rcf
import datetime

# local vars
cutoff = 0.5

def ensure_number(data):
    return np.nan_to_num(data)

def load_csv(file):
        #load data
        expression = []
        with open(file, "r") as csv_file:
            reader = csv.reader(csv_file, dialect='excel')
            for row in reader:
                expression.append(row)
        return expression

def load_descriptors(file):
        descriptors = []
        with open(file, "r") as tab_file:
            reader = csv.reader(tab_file, dialect='excel', delimiter='\t')
            descriptors = dict((rows[1],rows[2:]) for rows in reader)

        print('drug descriptors loaded. rows:  ' + str(len(descriptors)))
        return descriptors

def join_descriptors_label(expression,descriptors):
        unique_drugs = []
        # data set up
        data = []
        for row in expression:
            data.append(descriptors[row[0]])
            if row[0] not in unique_drugs:
                unique_drugs.append(row[0])
        data = np.array(data).astype(np.float32)

        labels = []
        for row in expression:
            labels.append(row[1:3])

        labels = np.array(labels).astype(np.float32)
        print('data size ' + str(len(data)) + ' labels size ' + str(len(labels)))
        return data,labels

def get_feature_dict(file, delimiter=',', key_index=0, use_int=False):
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=delimiter)
        next(reader)
        if use_int:
            my_dict = {}
            for row in reader:
                list = []
                for value in row[1:]:
                    list.append(int(value))
                my_dict[row[key_index]] = list
            return my_dict
        return dict((row[key_index], row[1:]) for row in reader)

def load_gene_expression_data(file,lm_gene_entrez_ids=None):
    return parse(
        file,
        col_meta_only=False, row_meta_only=False, rid=lm_gene_entrez_ids)

def printProgressBar (iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 25,
                      fill = '>', pct_interval=5):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    pct_iterations = max(int((total*pct_interval)/100), 1)
    if iteration % pct_iterations > 0:
        return
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s %s |%s| %s%% %s' % (datetime.datetime.now(), prefix, bar, percent, suffix), end = '\n')
    # Print New Line on Complete
    if iteration == total:
        print()

def get_trimmed_feature_dict(file, delimiter=',', key_index=0):
    keys_csv = load_csv(file)
    data = np.genfromtxt(file, delimiter=delimiter, skip_header=1)
    numerical = data[0:,1:]
    trimmed_data = rcf.trim_features(numerical)
    my_dict = {}
    for i in range(0, len(data)):
        my_dict[keys_csv[i+1][key_index]] = trimmed_data[i]
    return my_dict