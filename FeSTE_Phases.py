
import tensorflow as tf
from transformers import BertTokenizerFast
from tensorflow.python.framework import ops



from preprocessing import *

from model import create_cross_encoder, define_tensor_callbacks


def preliminaryFT(config_dict, entities_abstracts ,wgt_dir,current_run_test_dataset):

    "Split to sets"
    train_data, val_data, test_data = import_train_datasets(entities_abstracts, current_run_test_dataset)
    """Preprocess tuples to fit BERT input"""
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_feat_text, train_target_text, train_encoded_abstract, train_labels = lookup_domain_target_abs_to_cross_encoder_input(
        train_data[:20], tokenizer, only_abstratc_and_target=config_dict['only_abstratc_and_target'])
    val_feat_text, val_target_text, val_encoded_abstract, val_labels = lookup_domain_target_abs_to_cross_encoder_input(
        val_data[:20], tokenizer, only_abstratc_and_target=config_dict['only_abstratc_and_target'])

    train_labels_for_model = train_labels.to_numpy().reshape([-1, 1])
    val_labels_for_model = val_labels.to_numpy().reshape([-1, 1])

    """Create architecture"""
    model, embed_model = create_cross_encoder(learning_rate=config_dict['pre_lr'],
                                                  train_data_size=train_labels.shape[0], batch=config_dict['pre_batch'],
                                                  epochs=config_dict['pre_epochs'],
                                                  warmup_epochs=config_dict['pre_warmup_epochs'],
                                                  rateDacedy=config_dict["pre_rateDacedy"])

    tensorboard_callback, mcp_save, early_stoping = define_tensor_callbacks(results_dir=wgt_dir,
                                                                            metric=config_dict['early_stopping_metric'],
                                                                            patience=config_dict['specific_patience'],
                                                                            mode=config_dict['early_stopping_mode'])

    # train the model
    history = model.fit(train_encoded_abstract, train_labels_for_model, epochs=config_dict['pre_epochs'],
                          batch_size=config_dict['pre_batch'],
                          validation_data=(val_encoded_abstract, val_labels_for_model),
                          callbacks=[early_stoping, tensorboard_callback])

    model.save_weights(wgt_dir)
    print('Preliminary tuning weights been saved')
    del model, embed_model, train_encoded_abstract, train_labels_for_model, train_feat_text, train_target_text, train_labels
    ops.reset_default_graph()
    tf.keras.backend.clear_session()

def abstract_and_reformulation(current_dataset,entities_abstracts,func):
    datsets_names_tragets, dataset = create_combinations_of_each_lookup_target_from_func(func)
    abs_domain_data = datsets_names_tragets.merge(
        entities_abstracts[['dataset', 'lookup_x', 'abstract', 'lookup_lookup_domain', 'lookup_domain']],
        left_on=['dataset', current_dataset.lookup_col_name], right_on=['dataset', 'lookup_x'])
    current_dataset.abs_domain_dataset = abs_domain_data.copy()


def split_current_dataset(current_dataset, train_idx, test_idx, config_dict):
    current_dataset.create_train_test_folds_dfs(train_idx, test_idx)
    # merge with abstracts
    current_dataset.abs_feat_fold_df_train = current_dataset.create_train_set_for_bert(
        current_dataset.train_fold_df)
    config_dict['train_data_size'] = current_dataset.abs_feat_fold_df_train.shape[0]
    current_dataset.abs_feat_fold_df_test = current_dataset.create_train_set_for_bert(current_dataset.test_fold_df)
    current_dataset.abs_feat_fold_df = current_dataset.create_train_set_for_bert(current_dataset.original_data)


def prepare_data_for_bert_and_train(specific_encoder, abs_domain_dataset_train, abs_domain_dataset_val,
                                    num_of_trainig_ephocs, current_dataset,
                                    config_dict):
    """Creats e=the model, prepare the data, train the model, returns num of ephocs or the model itself"""

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_feat_text, train_target_text, train_encoded_abstract, train_labels = lookup_domain_target_abs_to_cross_encoder_input(
        abs_domain_dataset_train,
        tokenizer, lables_for_bert=current_dataset.col_label_for_bert,
        only_abstratc_and_target=config_dict['only_abstratc_and_target'])

    val_feat_text, val_target_text, val_encoded_abstract, val_labels = lookup_domain_target_abs_to_cross_encoder_input(
        abs_domain_dataset_val,
        tokenizer, lables_for_bert=current_dataset.col_label_for_bert,
        only_abstratc_and_target=config_dict['only_abstratc_and_target'])

    "train model with tarin / val split"
    train_encoded_abstract, train_labels_for_model = train_encoded_abstract, train_labels.to_numpy().reshape(
        [-1, 1])
    val_encoded_abstract, val_labels_for_model = val_encoded_abstract, val_labels.to_numpy().reshape(
        [-1, 1])

    _, _, early_stoping = define_tensor_callbacks(metric=config_dict['early_stopping_metric'],
                                                  patience=config_dict['specific_patience'],
                                                  mode=config_dict['early_stopping_mode'])

    specific_encoder.fit(train_encoded_abstract, train_labels_for_model, epochs=num_of_trainig_ephocs,
                         batch_size=config_dict['specific_batch'],
                         validation_data=(val_encoded_abstract, val_labels_for_model),
                         callbacks=[early_stoping])


def target_datasetFT(current_dataset, wgt_dir, config_dict):
    # split to train and val
    train_ratio = 0.5
    recs_num = config_dict['train_data_size']
    recs_idx = list(current_dataset.abs_feat_fold_df_train.index)
    print(recs_idx)
    np.random.seed(0)
    train_recs_idx = recs_idx[:int(
        recs_num * train_ratio)]  # np.random.choice(recs_idx,int(recs_num * train_ratio),replace=False)
    val_recs_idx = list(set(recs_idx) - set(train_recs_idx))
    abs_domain_dataset_train = current_dataset.abs_feat_fold_df_train.iloc[train_recs_idx, :]
    abs_domain_dataset_val = current_dataset.abs_feat_fold_df_train.iloc[val_recs_idx, :]
    """Create architecture"""
    specific_encoder, specific_embed_encoder = create_cross_encoder(learning_rate=config_dict['specific_lr'],
                                                                    train_data_size=config_dict[
                                                                        'train_data_size'],
                                                                    batch=config_dict['specific_batch'],
                                                                    epochs=config_dict['specific_epochs'],

                                                                    warmup_epochs=config_dict[
                                                                        'specific_warmup_epochs'],
                                                                    rateDacedy=0.0000,
                                                                    num_of_classes=current_dataset.num_of_classes_for_bert)

    specific_encoder.load_weights(wgt_dir)

    prepare_data_for_bert_and_train(specific_encoder,
                                    abs_domain_dataset_train,
                                    abs_domain_dataset_val,
                                    num_of_trainig_ephocs=config_dict['specific_epochs'],
                                    current_dataset=current_dataset,
                                    config_dict=config_dict)

    return specific_encoder, specific_embed_encoder



def predict_lables_likelihood(current_dataset, encoder,config_dict):
    """"recives dataset_encoded_abstract (list), split the list into batches of 1000 feed into the model,
     return the embeddings of each"""

    """Prepare datasets abstracts for prediction"""
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    _, _, dataset_encoded_abstract, _ = lookup_domain_target_abs_to_cross_encoder_input(
        current_dataset.abs_feat_fold_df, tokenizer=tokenizer,
        only_abstratc_and_target=config_dict['only_abstratc_and_target'])

    # use the pooled layer of each seq
    dataset_size = dataset_encoded_abstract[0].shape[0]
    num_of_embedded_recs = dataset_size
    recs_list = list(range(num_of_embedded_recs))
    start_pos = 0
    global_batch_size = 500
    current_dataset_embeds_np_ls = []
    while start_pos < dataset_size:
        end_pos = start_pos + global_batch_size

        if end_pos > dataset_size:
            end_pos = dataset_size

        print('start_pos:end_pos: ', start_pos, ':', end_pos)
        if len(dataset_encoded_abstract[1]) == 2:  # with features
            current_dataset_encoded_abstract = [dataset_encoded_abstract[0][start_pos:end_pos],
                                                [dataset_encoded_abstract[1][0][start_pos:end_pos],
                                                 dataset_encoded_abstract[1][1][start_pos:end_pos]]]
        else:  # only abstracts
            current_dataset_encoded_abstract = [dataset_encoded_abstract[0][start_pos:end_pos],
                                                dataset_encoded_abstract[1][start_pos:end_pos]]

            length_of_set = current_dataset_encoded_abstract[0].shape[0]

            current_dataset_embeds_np = encoder.predict(current_dataset_encoded_abstract, batch_size=1,
                                                        verbose=1).reshape(length_of_set)

        current_dataset_embeds_np_ls.append(current_dataset_embeds_np)
        del current_dataset_embeds_np, current_dataset_encoded_abstract

        start_pos = end_pos
    dataset_embeds_np = np.concatenate(current_dataset_embeds_np_ls, axis=0)
    return dataset_embeds_np

def transform_into_features(labels_prediction, current_dataset):
    current_dataset.abs_feat_fold_df['labels_pred'] = labels_prediction
    # apply softmax and make feature from each optional class
    lookups_target_with_bert_features_before_softmax = current_dataset.abs_feat_fold_df[
        [current_dataset.lookup_col_name, 'optional_target', 'labels_pred']].drop_duplicates(
        [current_dataset.lookup_col_name, 'optional_target']).pivot(index=current_dataset.lookup_col_name,
                                                                    columns="optional_target",
                                                                    values="labels_pred").reset_index()
    # if there are rcords without abstract
    if '' in lookups_target_with_bert_features_before_softmax.columns:
        lookups_target_with_bert_features_before_softmax = lookups_target_with_bert_features_before_softmax.drop(
            [''], 1)
    probabilities_for_softmax = lookups_target_with_bert_features_before_softmax.drop(
        [current_dataset.lookup_col_name], 1).to_numpy()
    # transform labels predictions to new features
    probabilities_after_softmax = np.apply_along_axis(softmax, 1, probabilities_for_softmax)
    probabilities_after_softmax_nan = np.nan_to_num(probabilities_after_softmax, nan=0)
    lookup_fetures_df_ls = [lookups_target_with_bert_features_before_softmax[current_dataset.lookup_col_name],
                            pd.DataFrame(probabilities_after_softmax_nan)]
    lookups_target_with_bert_features = pd.concat(lookup_fetures_df_ls, 1)

    """Merge probabilities with original lookups to have same contruct as orig dataset"""
    # create new features object
    current_dataset.entities_features = lookups_target_with_bert_features
    # merge with original dataset and make sure thre is no duplicates
    current_dataset.data['index'] = current_dataset.data.index

    current_dataset.new_features = current_dataset.data[[current_dataset.lookup_col_name, 'index']].merge(
        current_dataset.entities_features,
        how='left', left_on=[current_dataset.lookup_col_name],
        right_on=current_dataset.lookup_col_name).drop([current_dataset.lookup_col_name], 1)

    current_dataset.new_features = current_dataset.new_features.drop_duplicates('index').drop('index', 1)

    current_dataset.new_features = current_dataset.new_features.replace(np.nan, 0)
