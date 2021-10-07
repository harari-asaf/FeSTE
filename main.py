import json
import os
from sklearn.model_selection import StratifiedKFold
from FeSTE_Phases import*
from test_dataset_class import test_dataset
from metrics import  compute_acc_on_fold


def main():
    # available datasets (Only three for demonstration)
    dataset_to_run_on_dict = {1: 'aaup', 2: 'zoo', 3: 'analcatdata_reviewer'}
    test_dataset_id = 2
    current_run_test_dataset = dataset_to_run_on_dict[test_dataset_id]
    # import config
    config_dict = json.load(open('config.json'))
    super_dir = 'FeSTE'
    # folders for models wights
    print('super dir: ', super_dir)
    if not os.path.exists(super_dir):
        os.makedirs(super_dir)

    wgt_dir = super_dir + current_run_test_dataset + '/'
    if not os.path.exists(wgt_dir):
        os.makedirs(wgt_dir)
    print('domain dir: ', wgt_dir)

    # for demonstration we import pre-linked abstracts.
    # To execute entity linking phase use "query_google.py" to extract abstracts download DBpedias Dump file
    _, entities_abstracts = import_abstracts_df(abstracts_file_path='abstracts.csv')

    """Preliminary FT"""
    preliminaryFT(config_dict, entities_abstracts, wgt_dir,current_run_test_dataset)

    """Target dataset FT"""
    # import target dataset
    func = [func[1] for func in im.__dict__.items() if callable(func[1])][1:][test_dataset_id - 1]
    dataset_name, domain, dataset, _ = func()
    # create target dataset class
    current_dataset = test_dataset(dataset_name, domain, dataset)
    # add abstracts and reformulation
    abstract_and_reformulation(current_dataset, entities_abstracts, func)
    # split to folds
    sss = StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
    # only one fold for demonstration
    train_idx, test_idx = list(sss.split(current_dataset.X, current_dataset.y))[2]

    split_current_dataset(current_dataset, train_idx, test_idx,config_dict)
    """Target dataset FT"""
    specific_model, _ = target_datasetFT(current_dataset, wgt_dir, config_dict)
    """Features Generation"""
    labels_prediction = predict_lables_likelihood(current_dataset, encoder=specific_model,config_dict=config_dict)
    transform_into_features(labels_prediction, current_dataset)
    print('NEW FEATURES SET F_I^NEW: \n', current_dataset.new_features)
    """Evaluation"""
    cand_np = current_dataset.new_features.to_numpy()
    original_dataset_for_comparison = current_dataset.original_data.iloc[:, :-1]  # current_dataset.scaled_data_df#
    # Run classifiers and compute metrics
    compute_acc_on_fold(original_dataset_for_comparison, cand_np, train_idx,
                        test_idx)


if __name__ == '__main__':
    main()