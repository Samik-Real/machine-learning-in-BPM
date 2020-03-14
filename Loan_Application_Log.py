import sys
import time
from Case import *
import tensorflow as tf
import numpy as np
import random as rn


def main():
    seed = 11
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    rn.seed(seed)

    #This file  "BPI Challenge 2017.xes" needs to be exported from disco to get nice formatting and to program mining easier (just for the coder)
    file = 'Dataset/' + 'BPI Challenge 2017.xes'

    print("Started reading file: " + str(file))
    #Utility.logPrinter(file)

    case_hash_map, case_statistics = Utility.logReader(file)
    print(case_statistics)
    time.sleep(0.5)

    print(case_hash_map.get('Application_652823628').toString())
    print(case_hash_map.get('Application_1413308979').toString())
    time.sleep(0.5)

    file_name_1 = 'Dataset/' + 'BPI_Challenge_Loan_Application.csv'
    CSV.writeCSV(hashmap=case_hash_map, fileName=file_name_1, earliestTimeStamp=datetime(2016,1,1,10,51,15))

    one_hot_encoder = CSV.One_Hot_Encoder(file=file_name_1, attribute_names_to_encode=['Application Type', 'Loan Goal'])
    machine_learning_ready_csv_file_name = one_hot_encoder.fit_data_to_oneHot_and_save_CSV()

    # Csv with expert rate by adding column to csv
    expert_rate_csv = pd.read_csv('Dataset/with_expert_rate.csv')
    data_frame = pd.read_csv(machine_learning_ready_csv_file_name)

    data_frame = pd.concat([data_frame, expert_rate_csv], axis=1)
    data_frame.to_csv(machine_learning_ready_csv_file_name, index=False)





    # Filter Feature Selection
    string = 'filter_featureSelection_3'
    selected_features = ''
    y = ''
    filter_feature_selection = FeatureSelection.Filter(file=machine_learning_ready_csv_file_name)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Correlation Matrix: \n" + str(filter_feature_selection.correlationMatrix()))


    num_of_top_values = 'all'
    if string == 'filter_featureSelection_1':
        y = ['Application Status']
        x = ['Case ID', 'Application Status', 'Offer Selected', 'Offered Amount', 'First Withdrawal Amount']  # attributes with correlation > 0.5 are removed
        selected_features = filter_feature_selection.spearmansCorrelation(x_columns=x, y_column=y, remove_x_columns=True, num_of_top_values=num_of_top_values)
    elif string == 'filter_featureSelection_2':
        y = ['Offer Selected']
        x = ['Case ID', 'Offer Selected', 'Application Status', 'First Withdrawal Amount', 'Offered Amount']  # Offered amount is removed to see what else decides if offer is selected
        selected_features = filter_feature_selection.spearmansCorrelation(x_columns=x, y_column=y, remove_x_columns=True, num_of_top_values=num_of_top_values)
    elif string == 'filter_featureSelection_3':
        y = ['On-time']
        x = ['Case ID', 'On-time', 'Total Case Duration', 'Application Status', 'Normalized Handovers'] # attributes with correlation > 0.55 are removed
        selected_features = filter_feature_selection.spearmansCorrelation(x_columns=x, y_column=y, remove_x_columns=True, num_of_top_values=num_of_top_values)
    else:
        raise Exception("Feature Selection attributes not understood")


    # Wrapper Feature Selection
    wrapper_feature_selection = FeatureSelection.Wrapper(file=machine_learning_ready_csv_file_name)
    #wrapper_feature_selection.recursiveFeatureElimination(x_columns=selected_features, y_column=y, number_of_features=1)
    selected_features = wrapper_feature_selection.recursiveFeatureEliminationWithCrossValidation(x_columns=selected_features, y_column=y, cross_validation=10)
    print("Selected Features: " + str(selected_features))
    #wrapper_feature_selection.sequentialFeatureSelector(x_columns=selected_features, y_column=y)


    model_type = 'decision_tree'
    #model_type = 'random_forrest'
    #model_type = 'multilayer_perceptron'
    classifier = Classification(file=machine_learning_ready_csv_file_name, class_type=y[0])
    classifier.classify(x_columns=selected_features, y_column=y, remove_x_columns=False, classification_type=model_type)

if __name__ == '__main__':
    main()