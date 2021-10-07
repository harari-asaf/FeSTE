import pandas as pd


def lookup_front_target_end(dataset,name_for_lookup, name_for_target):
    lookup = dataset['lookup']
    traget = dataset['target']
    dataset = dataset.drop(['lookup', 'target'], axis=1)
    dataset.insert(0, name_for_lookup, lookup)
    dataset.insert(len(dataset.columns), name_for_target, traget)
    return dataset

def import_aaup():
    """import and preper  aaup dataset
     https://www.openml.org/d/488"""
    description = """A data set about universities contains information on salaries and attributes of 1161 American colleges and universities. The task is to classify the universities to low, medium, high by salary"""
    dataset_name = 'aaup'
    lookup_domain = 'university'

    aaup = pd.read_csv('Datasets/aaup.csv',header=None)
    # only Federal ID number, College name, State , Type  , Average salary - all ranks, Number of pro/assitents etc
    # 7 is the target.
    aaup = aaup[[0,1,2,3,7,12,13,14,15,16]]
    aaup = aaup.rename(columns = {1:'lookup', 7:'target'})
    aaup['target'] = pd.qcut(aaup['target'], q=3, precision=0,labels = ['low','medium','high'])
    """target to the end, lookup to front and names"""
    aaup = lookup_front_target_end(aaup, name_for_lookup=lookup_domain, name_for_target='salaries')

    aaup.head()
    dataset = aaup

    return dataset_name, lookup_domain, dataset, description

def import_zoo():
    description = """A dataset about Zoology containing 17 boolean valued attributes describing animals. The task is to classify the animals to types: Mammals, Fishes etc"""

    lookup_domain = 'animal'
    dataset_name = 'zoo'
    zoo = pd.read_csv('Datasets/zoo.csv',header=None)
    zoo = zoo.rename(columns = {0:'lookup', 17:'target'})

    """target to the end, lookup to front and names"""
    zoo = lookup_front_target_end(zoo,name_for_lookup = lookup_domain, name_for_target = 'class')

    """Change target values"""
    zoo['class'] = zoo['class'].replace({1:'Mammal',2:'Aves',3:'Reptile',4:'Fish',5:'Amphibian',6:'Insect',7:'Malacostraca'})

    zoo.head()
    dataset = zoo

    return dataset_name, lookup_domain, dataset, description


def import_analcatdata_reviewer():
    description = """A dataset about Movies, contains information of movies ranking by filim critics. 
        The task is multiclass classification, to predict the rank of the movie by Roger Ebert."""
    def strip(field):
        field = field.strip()
        return field

    dataset_name = 'analcatdata_reviewer'
    dataset = pd.read_csv('Datasets/'+dataset_name+'.csv',converters={'lookup':strip})
    dataset.head()
    dataset = dataset.rename(columns={'Film': 'lookup', 'Roger_Ebert': 'target'})
    'remove records with missing values in the target'
    dataset = dataset.loc[dataset.target != '?', :]
    lookup_domain = 'movie'

    dataset['target'] = dataset['target'].replace({'Con': 'negative', 'Pro': 'positive'})
    """target to the end, lookup to front and names"""
    dataset = lookup_front_target_end(dataset, name_for_lookup=lookup_domain, name_for_target='Roger Ebert rating')

    return dataset_name, lookup_domain, dataset, description



