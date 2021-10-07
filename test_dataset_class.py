import pandas as pd
import numpy as np


class test_dataset:

    def __init__(self, dataset_name, domain, dataset):
        self.name = dataset_name.capitalize()
        self.domain = domain.capitalize()
        self.original_data = dataset.copy()
        self.num_of_feat =  dataset.shape[1] -2

        self.target_col_name = dataset.columns[-1]
        self.lookup_col_name = dataset.columns[0]
        self.feat_col_names = list(dataset.drop([self.target_col_name,self.lookup_col_name],1).columns)
        self.num_of_classes = len(dataset.iloc[:,-1].unique())
        self.set = ''
        self.unique_entities = ''
        self.X = np.array(pd.get_dummies(self.original_data))
        self.y = np.array(self.original_data.loc[:, self.target_col_name])
        self.scaled_feat_col_names = ''
        # data that can be modified
        self.data = dataset.copy()
        self.col_label_for_bert = self.data.columns[-1]
        self.data['dataset'] = self.name
        self.abs_domain_dataset = ''
        self.new_features=''
        self.scaler = ''
        self.OneHotEncoder = ''



    def create_train_test_folds_dfs(self, train_idx, test_idx):
        self.train_fold_df = self.original_data.iloc[train_idx, :].reset_index(drop=True)
        self.test_fold_df = self.original_data.iloc[test_idx, :].reset_index(drop=True)




    def create_train_set_for_bert(self,set):
        # merge_original dataset with abstracts by lookup
        set['key'] = set.index

        abs_feat_fold_df = set.merge(self.abs_domain_dataset, how='left',
                                     on=[self.lookup_col_name]).drop_duplicates(['key','optional_target']).reset_index().drop('key', 1)

        abs_feat_fold_df = abs_feat_fold_df.rename(
            columns={self.target_col_name + '_x': self.target_col_name})

        abs_feat_fold_df = abs_feat_fold_df.replace(np.nan, '')

        abs_feat_fold_df.label = (abs_feat_fold_df.target == abs_feat_fold_df.optional_target)*1
        self.col_label_for_bert = 'label'

        self.num_of_classes_for_bert = len(abs_feat_fold_df.loc[:, self.col_label_for_bert].unique())
        # self.train_data_size = abs_feat_fold_df.shape[0]
        return abs_feat_fold_df



