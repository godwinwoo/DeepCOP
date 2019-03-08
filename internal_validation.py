import datetime
import numpy as np
from Helpers.callbacks import NEpochLogger
from Helpers.utilities import all_stats
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, EarlyStopping
from sklearn.model_selection import train_test_split, KFold
import sklearn.metrics as metrics
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose the gpu to use in multi gpu system


def save_model(model, file_prefix, run_number):
    file_name = file_prefix + '_' + str(run_number)
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_name + ".h5")
    print("Saved model", file_name)


def get_model(n_classes, neuron_count):
    dropout = 0.2
    activation_input = 'selu'
    activation = 'relu'
    activation_output = 'softmax'

    model = Sequential()
    # input and first layer
    model.add(Dense(neuron_count, input_shape=(neuron_count,)))
    model.add(BatchNormalization())
    model.add(Activation(activation_input))
    model.add(Dropout(dropout))

    # 2nd layer
    model.add(Dense(neuron_count))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # output later
    model.add(Dense(n_classes))
    model.add(BatchNormalization())
    model.add(Activation(activation_output))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def do_validation(data, labels, model_file_prefix):
    nb_classes = 2
    d = data.shape[1]
    neuron_count = int(d)
    nb_epoch = 10000
    n_splits = 10
    batch_size = 2**12

    labels = np_utils.to_categorical(labels, nb_classes)
    kf = KFold(n_splits=n_splits, shuffle=True)
    sum_auc = 0
    count = 0
    sum_prec = 0
    sum_fscore = 0
    sum_ef = 0

    for train_indexes, test_indexes in kf.split(data):
        count += 1
        print("TRAIN:", train_indexes, "TEST:", test_indexes)

        # take half of the test data as validation
        X_train = data[train_indexes]
        Y_train = labels[train_indexes]
        val_indexes, test_indexes = train_test_split(test_indexes, train_size=0.5, test_size=0.5, shuffle=True)
        X_val = data[val_indexes]
        Y_val = labels[val_indexes]
        X_test = data[test_indexes]
        Y_test = labels[test_indexes]

        model = get_model(nb_classes, neuron_count)

        # train the model
        history = History()
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        out_epoch = NEpochLogger(display=5)
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=0, validation_data=(X_val, Y_val), callbacks=[history, early_stopping, out_epoch],
                  class_weight='auto')
        save_model(model, model_file_prefix, count)

        # get results and report
        y_pred_train = model.predict_proba(X_train)
        y_pred_test = model.predict_proba(X_test)

        # report accuracy
        y_pred = np.argmax(y_pred_test, axis=1)
        y_true = np.argmax(Y_test, axis=1)
        report = metrics.classification_report(y_true, y_pred)
        print("Test Report", report)

        # report auc
        train_stats = all_stats(Y_train[:, 1], y_pred_train[:, 1])
        test_stats = all_stats(Y_test[:, 1], y_pred_test[:, 1])
        print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff | Max F score')
        print('All stats train:', ['{:6.3f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.3f}'.format(val) for val in test_stats])

        # get enrichment factor
        precision = float(test_stats[4])
        tokens = report.split()
        support = int(tokens[13])
        total = int(tokens[20])
        ef = precision / (support/total)

        # get average
        sum_auc += test_stats[0]
        sum_prec += test_stats[4]
        sum_fscore += test_stats[6]
        sum_ef += ef

        print("running kfold auc", count, sum_auc / count,
              'prec', sum_prec / count,
              'fscore', sum_fscore / count,
              'enrichment', sum_ef / count)

def load_and_validate():
    # target_cell_names = ['VCAP', 'A549', 'A375', 'PC3', 'MCF7', 'HT29', 'LNCAP']
    target_cell_names = ['LNCAP']  # choose the cell line(s) to do x10

    load_data_folder_path = "TrainData/"
    save_models_folder_path = "SavedModels/"
    percentiles = [5]
    for target_cell_name in target_cell_names:
        for percentile_down in percentiles:
            file_suffix = target_cell_name + '_' + str(percentile_down) + 'p'
            npX = np.load(load_data_folder_path + file_suffix + "_X.npz")['arr_0']
            for direction in ["Down", "Up"]:
                file_suffix = target_cell_name + '_' + direction + '_' + str(percentile_down) + 'p'
                model_file_prefix = save_models_folder_path + file_suffix
                print('load location', load_data_folder_path)
                print('save location', model_file_prefix)
                npY_class = np.load(load_data_folder_path + file_suffix + "_Y_class.npz")['arr_0']
                do_validation(npX, npY_class, model_file_prefix)

load_and_validate()