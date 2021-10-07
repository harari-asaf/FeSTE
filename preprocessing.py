import pandas as pd
import import_datasets as im
import numpy as np
import regex as re
def preper_retrived_entities_data(data_df_10_results):
    """Remove nulls and disambiguation entities
    Domains and datastes names to be capitilized
    return DF"""
    # remove nulls
    data_df_10_results = data_df_10_results.loc[~data_df_10_results.abstract.isna(),:]
    # remove disambuation pages
    data_df_10_results = data_df_10_results.loc[~data_df_10_results.abstract.str.contains('may refer to') , :].reset_index(drop=True)
    # change domains names to be conssitence (upper case)
    data_df_10_results.lookup_domain = data_df_10_results.lookup_domain.apply(lambda x: x.capitalize())
    data_df_10_results.dataset = data_df_10_results.dataset.apply(lambda x: x.capitalize())

    return data_df_10_results

def import_abstracts_df( abstracts_file_path):
    data_df_10_results = pd.read_csv(abstracts_file_path)

    # preprocee file
    data_df_10_results = preper_retrived_entities_data(data_df_10_results)
    # use only first entity
    data_df_first_result = data_df_10_results.drop_duplicates(['lookup_x', 'lookup_domain', 'dataset'])

    return data_df_10_results, data_df_first_result


def create_combinations_of_each_lookup_target(datsets_names_tragets):
    unique_targets = pd.DataFrame(datsets_names_tragets.target.unique(), columns=['optional_target'])
    unique_targets['key'] = 1
    datsets_names_tragets['key'] = 1
    datsets_names_tragets = pd.merge(datsets_names_tragets, unique_targets, on='key').drop('key', 1)
    # add label for bert
    datsets_names_tragets['label'] = (datsets_names_tragets.optional_target == datsets_names_tragets.target) * 1
    datsets_names_tragets['target_domain_optional_target'] = datsets_names_tragets['target_domain'] + ' is ' + \
                                                             datsets_names_tragets['optional_target']

    return datsets_names_tragets



def create_domain_dataset_target_domain_cols(func):
    dataset_name, domain, dataset, description = func()

    domain = domain.capitalize()
    dataset_name = dataset_name.capitalize()
    target_domain = dataset.columns[-1]
    dataset['domain'] = domain
    dataset['dataset'] = dataset_name
    dataset['target_domain'] = target_domain
    dataset[target_domain] = dataset[target_domain].astype('str')

    lookup_name = dataset.columns[0]
    datsets_names_tragets = dataset[['dataset', 'target_domain', 'domain']]
    datsets_names_tragets['target'] = dataset.loc[:, target_domain]
    datsets_names_tragets['lookup'] = dataset.iloc[:, 0]
    datsets_names_tragets[dataset.columns[0]] = datsets_names_tragets['lookup']
    datsets_names_tragets['orig_idx'] = dataset.index
    return datsets_names_tragets, dataset

def create_combinations_of_each_lookup_target_from_func(func):
    """create combinations of each lookup-target value for training"""

    datsets_names_tragets, dataset = create_domain_dataset_target_domain_cols(func)

    datsets_names_tragets = create_combinations_of_each_lookup_target(datsets_names_tragets)


    return datsets_names_tragets, dataset

def create_dataset_df():
    """import datasets data and create data for trainig:
    for sentense classification: create combinations of each record and optional target"""
    datasets_import_func = [func[1] for func in im.__dict__.items() if callable(func[1])][1:]
    print('number of datasets:', len(datasets_import_func))

    # datasets_not_in_analysis = ['WHO_Deaths_HIV', 'MetacriticAlbums', 'movies', 'books', 'Cities']

    datasets_list = []
    set_datasets_domains_ls = []
    datsets_names_tragets_ls = []
    descriptions_ls = []
    dataset_name_ls = []
    dataset_traget_names_ls = []
    for func in datasets_import_func:
        dataset_name, domain, dataset, description = func()


        descriptions_ls.append(description)
        dataset_name_ls.append(dataset_name)
        dataset_traget_names_ls.append(dataset.columns[-1])
        """create combinations of each lookup-target value for training"""
        datsets_names_tragets, dataset = create_combinations_of_each_lookup_target_from_func(func)
        # datsets_names_tragets = create_combinations_of_each_lookup_target(dataset, target_domain)


        # save to lists
        datsets_names_tragets_ls.append(datsets_names_tragets)
        datasets_list.append(dataset)
        set_datasets_domains_ls.append(domain)

    datsets_names_tragets_df = pd.concat(datsets_names_tragets_ls)


    return datasets_list, datsets_names_tragets_df, [descriptions_ls,dataset_name_ls,dataset_traget_names_ls]


    """preprocessing"""
def import_train_datasets(data_df_first_result,current_run_test_dataset):
    datasets_ls, datsets_names_tragets_df, description_ls = create_dataset_df()

    abs_domain_data = datsets_names_tragets_df.merge(
        data_df_first_result[['dataset', 'lookup_x', 'abstract', 'lookup_lookup_domain', 'lookup_domain']],
        left_on=['dataset', 'lookup', ], right_on=['dataset', 'lookup_x'])
    test_data = abs_domain_data.loc[abs_domain_data.dataset == current_run_test_dataset.capitalize(),
                :]  # abs_domain_data.loc[abs_domain_data.domain.isin(test_domain_ls),:]
    not_test_data = abs_domain_data.loc[abs_domain_data.dataset != current_run_test_dataset.capitalize(),
                    :]  # abs_domain_data.loc[abs_domain_data.domain.isin(train_domains_ls), :]

    print('replicate small datasets!!')
    dataset_value_counts = not_test_data.value_counts('dataset')
    datasets_with_less_1000 = dataset_value_counts.loc[dataset_value_counts < 1000].index
    dataset_value_counts.loc[dataset_value_counts < 1000]
    data_to_replicate = not_test_data.loc[not_test_data.dataset.isin(datasets_with_less_1000), :]
    not_test_data = pd.concat([not_test_data, data_to_replicate, data_to_replicate]).reset_index(drop=False)

    """Split to train/val"""
    unique_entities_in_data = not_test_data['lookup'].unique()
    num_of_entities = len(unique_entities_in_data)
    train_ratio = 0.8
    np.random.seed(30)
    train_entities = np.random.choice(unique_entities_in_data, int(train_ratio * num_of_entities), replace=False)
    train_data = not_test_data.loc[not_test_data['lookup'].isin(train_entities), :]
    val_data = not_test_data.loc[~not_test_data['lookup'].isin(train_entities), :]
    return train_data, val_data, test_data


def preper_text_to_bert(text_ls):
    # remove text between ()
    text_ls = [re.sub("[\(\[].*?[\)\]]", "", text) for text in text_ls]
    # remove special chars and Punctuation

    prepered_text_ls = [ re.sub('[^A-Za-z0-9 ]+', '', text.replace('_',' ').lower()) for text in text_ls]
    return prepered_text_ls



def lookup_domain_target_abs_to_cross_encoder_input(data_df, tokenizer, lables_for_bert='label', only_abstratc_and_target = 'all'):
    """only_abstratc_and_target : 'abstract_and_target'#'abstract'#'all'"""
    lookup_lookup_domain = preper_text_to_bert(data_df.lookup_lookup_domain.to_list())

    abstract = preper_text_to_bert(data_df.abstract.to_list())

    labels = data_df[lables_for_bert]

    if only_abstratc_and_target == 'abstract':
        target_sentence = ''
        encoded_lookup_domain_abstact = tokenizer(abstract, truncation=True, padding='max_length')
    elif only_abstratc_and_target == 'abstract_and_target': # dont use knowladge about domain
        target_sentence = preper_text_to_bert(data_df.optional_target.to_list())
        print('target senteces: \n', target_sentence[:10])
        encoded_lookup_domain_abstact = tokenizer(abstract,target_sentence, truncation=True, padding='max_length')
    else:
        target_domain = preper_text_to_bert(data_df.target_domain.to_list())
        target_domain_optional_target = preper_text_to_bert(data_df.target_domain_optional_target.to_list())
        target_sentence = target_domain_optional_target
        print('target senteces: \n', target_sentence[:10])
        encoded_lookup_domain_abstact = tokenizer(abstract,target_sentence, truncation=True, padding='max_length' )
    # encoded_lookup_domain_abstact = [np.array(encoded_lookup_domain_abstact['input_ids']),np.array(encoded_lookup_domain_abstact['attention_mask'])]
    cross_encoder_input = [np.array(encoded_lookup_domain_abstact['input_ids']), np.array(encoded_lookup_domain_abstact['attention_mask'])]



    return  abstract, target_sentence, cross_encoder_input, labels

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()