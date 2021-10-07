import import_datasets as im
import pandas as pd
import numpy as np
from googlesearch import search
import os


def query_using_google(lookup,domain,target_domain):
    query_google = lookup + " " + domain + ' '+ target_domain#+ " site:en.wikipedia.org "
    print(query_google)
    lookup_urls = [i for i in search(query_google, tld='com', lang='en', num=50, stop=50, pause=0)]
    if len(lookup_urls)>0:

        return lookup_urls
    else: return [np.nan]



data_output_folder = 'linkedData/' # 'entity_matching_distil/retrival_info/'

if not os.path.exists(data_output_folder):
    os.mkdir(data_output_folder)


datasets_and_domains = []
datasets_import_func = [func[1] for func in im.__dict__.items() if  callable(func[1])][1:]
print('number of datasets:',len(datasets_import_func))


# iterate over import functions (of each dataset)
for func in datasets_import_func:
    error_in_lookup = -1

    retrieval_results = []
    first_retrieval_result = []
    num_of_searches = 0
    i = 0

    dataset_name, domain, dataset , description= func()
    target_domain = dataset.columns[-1]
    print('DATASET: ', dataset_name)

    info_output_file = dataset_name + '.csv'
    # check number of unique entities
    unique_lookups = dataset.iloc[:, 0].unique()
    print('Number of unique lookup in dataset: ', len(unique_lookups))


    # iterate over unique entities column
    while i<len(unique_lookups):
            print(i)
            lookup = unique_lookups[i]
            print(lookup)
            num_of_searches += 1
            # google_entities = query_using_google(lookup, domain)
            google_entities = query_using_google(lookup, domain, target_domain )
            retrieval_results.append(google_entities)
            first_retrieval_result.append(google_entities[0])
            i += 1



    # save to df
    retrieval_results_df = pd.DataFrame(list(zip(unique_lookups, first_retrieval_result, retrieval_results)),columns = ['lookup','first_entity','entities'])

    """ Save lookups and entities with all data"""
    # Save dataset info for further analysis
    retrieval_results_df['dataset'] = dataset_name
    retrieval_results_df['dataset_description'] = description
    retrieval_results_df['lookup_domain'] = domain

    retrieval_results_df.to_csv(data_output_folder+info_output_file, index=False)


