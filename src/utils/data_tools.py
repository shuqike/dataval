"""Refer to https://github.com/ykwon0407/dataoob/blob/main/dataoob/preprocess/utils.py"""
import os, pickle, argparse, warnings
import numpy as np
import pandas as pd
import torch
import openml


class CustomDataset(torch.utils.data.Dataset):
    """Refer to https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {
            'features': self.X[idx],
            'labels': torch.tensor(self.y[idx]),
        }
        return item


def create_dataset(X, y):
    return CustomDataset(X, y)

def create_df_openml_datasets(n_minimum=5*1e3, n_maximum=1e5, p_maximum=1e3):
    """Return pandas.DataFrame of available datasets such that 
    - no missing values
    - 5*1e3 <= sample size <= 1e5
    - input dimension <= 1e3
    """
    df_every_datasets = openml.datasets.list_datasets(output_format="dataframe")

    result = []
    for index, row in df_every_datasets.iterrows():
        if (row['NumberOfInstancesWithMissingValues'] != 0) or (row['NumberOfMissingValues'] != 0):
            continue
        if (row['NumberOfInstances'] < n_minimum) or (row['NumberOfInstances'] > n_maximum):
            continue
        if row['NumberOfFeatures'] > p_maximum:
            continue
        
        if row['NumberOfClasses'] >= 2:
            result.append([row['did'], row['name'], 'clf'])
        elif row['NumberOfClasses'] == 0:
            result.append([row['did'], row['name'], 'reg'])
        else:
            pass

    df_openml_datasets = pd.DataFrame(result, columns=['dataset_id', 'name', 'task_type'])
    return df_openml_datasets

def dataset_maker(dataset_num, name, task_type):
    def judge(X_col):
        for i in X_col.index.values:
            if X_col[i] is not None:
                return isinstance(X_col[i], str)
        return False

    save_dict = {}
    try:
        dataset = openml.datasets.get_dataset(dataset_num)
    except:
        return {'dataset_name':'no name'}, 'no name', 'cannot get dataset', 'no name'
    save_dict['description'] = dataset.description
    save_dict['dataset_name'] = name
    file_name = name+'_'+str(dataset_num)+'.pkl'
    
    try:
        X, y, categorical_indicator, attribute_info = dataset.get_data(target=dataset.default_target_attribute,
                                                                       dataset_format="dataframe")
    except:
        return save_dict, file_name, 'attribute error because of openml', save_dict['dataset_name']
    
    if task_type in ['clf']:
        list_of_classes, y = np.unique(y, return_inverse=True)
        
    if y is None:
        return save_dict, file_name, 'y is none', save_dict['dataset_name']
    
    row_num = X.shape[0] 
    y = np.array(y)
    
    save_dict['y'] = y
    save_dict['flag'] = -1000
    save_dict['num_names'] = []
    save_dict['cat_names'] = []
    save_dict['X_num'] = []
    save_dict['X_cat'] = []
    save_dict['n_samples'] = row_num
    missing_val = 0
    for name_ind, name in enumerate(attribute_info):
        if name in ['ID','url']:
            continue
            
        if (categorical_indicator[name_ind]) or (judge(X[name])):
            save_dict['cat_names'].append(attribute_info[name_ind])
            list_of_classes, now = np.unique(X[name].astype(str), return_inverse=True)
            for j_ind, j in enumerate(X[name].index.values):
                if isinstance(X[name][j], float):
                    now[j_ind] = save_dict['flag']
                    missing_val += 1
            save_dict['X_cat'].append(now)
        else:
            save_dict['num_names'].append(attribute_info[name_ind])
            save_dict['X_num'].append(np.array(X[name], dtype=float))
            
    save_dict['X_num'] = np.array(save_dict['X_num']).transpose()
    save_dict['X_cat'] = np.array(save_dict['X_cat']).transpose()
    save_dict['missing_value'] = len(save_dict['X_num'][np.isnan(save_dict['X_num'])]) + len(save_dict['X_cat'][np.isnan(save_dict['X_cat'])]) + missing_val
    save_dict['X_num'][np.isnan(save_dict['X_num'])] = save_dict['flag']
    save_dict['X_cat'][np.isnan(save_dict['X_cat'])] = save_dict['flag']
    
    return save_dict, file_name, 'success', save_dict['dataset_name']

def download_openML_dataset(path='../data/', n_minimum=5*1e3, n_maximum=1e5, p_maximum=1e3, use_sample=True):
    openml.config.set_cache_directory(path)
    if use_sample is True:
        import pandas as pd
        print('Read dataset information from sample_openml.csv')
        df_openml_datasets = pd.read_csv(f'{path}/sample_openml.csv')
    else:
        warnings.warn('It would take a bit of time and memory. You may want to check --use_sample')
        df_openml_datasets = create_df_openml_datasets(
            n_minimum=n_minimum,
            n_maximum=n_maximum, 
            p_maximum=p_maximum
        )

    # make necessary directories
    if not os.path.exists(f'{path}/dataset_clf_openml'):
        print(f'In {path}, dataset_clf_openml does not exist.')
        os.makedirs(f'{path}/dataset_clf_openml')

    # make necessary directories
    for index, row in df_openml_datasets.iterrows():
        dataset_num, name, task_type = row['dataset_id'], row['name'], row['task_type']
        file_name = name+'_'+str(dataset_num)+'.pkl'

        if not os.path.exists(f'{path}/{file_name}'):
            print(f'{file_name} does not exist. Will download it from openml', flush=True)
            save_dict, file_name, indication, dataset_name = dataset_maker(dataset_num, name, task_type) 
            if indication == 'success':
                if not os.path.exists(f'{path}/dataset_{task_type}_openml/{file_name}'):
                    with open(f'{path}/dataset_{task_type}_openml/{file_name}', 'wb') as handle:
                        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                n_samples=save_dict['n_samples']
                n_variables=len(save_dict['num_names'])+len(save_dict['cat_names'])
                print(f'ID: {dataset_num:<5}, N: {n_samples:<5}, p: {n_variables:<5}, Name: {dataset_name:<30}')
            else:
                print('Indication is not success', dataset_num, dataset_name, indication)
            del save_dict, file_name, indication
