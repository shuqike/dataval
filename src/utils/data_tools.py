"""Refer to https://github.com/ykwon0407/dataoob/"""
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

def load_data(problem, dataset, **dargs):
    print('-'*30)
    print(dargs)
    if problem=='clf':
        (X, y), (X_val, y_val), (X_test, y_test)=load_classification_dataset(dataset=dataset,
                                                                            n_data_to_be_valued=dargs['n_data_to_be_valued'],
                                                                            n_val=dargs['n_val'],
                                                                            n_test=dargs['n_test'],
                                                                            input_dim=dargs.get('input_dim', 10),
                                                                            clf_path=dargs.get('clf_path'),
                                                                            openml_path=dargs.get('openml_clf_path'))
        if dargs['is_noisy'] > 0:
            n_class=len(np.unique(y))

            # training is flipped
            flipped_index=np.random.choice(np.arange(dargs['n_data_to_be_valued']), 
                                           int(dargs['n_data_to_be_valued']*dargs['is_noisy']), 
                                           replace=False) 
            random_shift=np.random.choice(n_class-1, len(flipped_index), replace=True)
            y[flipped_index]=(y[flipped_index] + 1 + random_shift) % n_class

            # validation is also flipped
            flipped_val_index=np.random.choice(np.arange(dargs['n_val']),
                                               int(dargs['n_val']*dargs['is_noisy']), 
                                               replace=False) 
            random_shift=np.random.choice(n_class-1, len(flipped_val_index), replace=True)
            y_val[flipped_val_index]=(y_val[flipped_val_index] + 1 + random_shift) % n_class 
            return (X, y), (X_val, y_val), (X_test, y_test), flipped_index
        else:
            return (X, y), (X_val, y_val), (X_test, y_test), None
    else:
        raise NotImplementedError('Check problem')

def load_classification_dataset(dataset,
                                n_data_to_be_valued, 
                                n_val, 
                                n_test, 
                                input_dim=10,
                                clf_path='clf_path',
                                openml_path='openml_path'):
    '''
    This function loads classification datasets.
    n_data_to_be_valued: The number of data points to be valued.
    n_val: Validation size. Validation dataset is used to evalute utility function.
    n_test: Test size. Test dataset is used to evalute model performance.
    clf_path: path to classification datasets.
    openml_path: path to openml datasets.
    '''
    if dataset == 'gaussian':
        print('-'*50)
        print('GAUSSIAN-C')
        print('-'*50)
        n, input_dim=max(100000, n_data_to_be_valued+n_val+n_test+1), input_dim
        data = np.random.normal(size=(n,input_dim))
        # beta_true = np.array([2.0, 1.0, 0.0, 0.0, 0.0]).reshape(input_dim,1)
        beta_true = np.random.normal(size=input_dim).reshape(input_dim,1)
        p_true = np.exp(data.dot(beta_true))/(1.+np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'pol':
        print('-'*50)
        print('pol')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/pol_722.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'jannis':
        print('-'*50)
        print('jannis')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/jannis_43977.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'lawschool':
        print('-'*50)
        print('law-school-admission-bianry')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/law-school-admission-bianry_43890.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'fried':
        print('-'*50)
        print('fried')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/fried_901.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'vehicle_sensIT':
        print('-'*50)
        print('vehicle_sensIT')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/vehicle_sensIT_357.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'electricity':
        print('-'*50)
        print('electricity')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/electricity_44080.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == '2dplanes':
        print('-'*50)
        print('2dplanes_727')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/2dplanes_727.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'creditcard':
        print('-'*50)
        print('creditcard')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/default-of-credit-card-clients_42477.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'covertype':
        print('-'*50)
        print('Covertype')
        print('-'*50)
        from sklearn.datasets import fetch_covtype
        data, target=fetch_covtype(data_home=clf_path, return_X_y=True)
    elif dataset == 'nomao':
        print('-'*50)
        print('nomao')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/nomao_1486.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'webdata_wXa':
        print('-'*50)
        print('webdata_wXa')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/webdata_wXa_350.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'MiniBooNE':
        print('-'*50)
        print('MiniBooNE')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/MiniBooNE_43974.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']         
    elif dataset == 'fabert':
        print('-'*50)
        print('fabert')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/fabert_41164.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'gas_drift':
        print('-'*50)
        print('gas_drift')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/gas-drift_1476.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'har':
        print('-'*50)
        print('har')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/har_1478.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'musk':
        print('-'*50)
        print('musk')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/musk_1116.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']     
    elif dataset == 'magictelescope':
        print('-'*50)
        print('MagicTelescope')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/MagicTelescope_44073.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'ailerons':
        print('-'*50)
        print('ailerons')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/ailerons_734.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'eye_movements':
        print('-'*50)
        print('eye_movements')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/eye_movements_1044.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'pendigits':
        print('-'*50)
        print('pendigits')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/pendigits_1019.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    else:
        assert False, f"Check {dataset}"

    (X, y), (X_val, y_val), (X_test, y_test) = preprocess_and_split_dataset(data, target,  n_data_to_be_valued, n_val, n_test)   

    return (X, y), (X_val, y_val), (X_test, y_test) 

def preprocess_and_split_dataset(data, target, n_data_to_be_valued, n_val, n_test, is_classification=True):
    if is_classification is True:
        # classification
        target = target.astype(np.int32)
    else:
        # regression
        target_mean, target_std= np.mean(target, 0), np.std(target, 0)
        target = (target - target_mean) / np.clip(target_std, 1e-12, None)
    
    ind=np.random.permutation(len(data))
    data, target=data[ind], target[ind]

    data_mean, data_std= np.mean(data, 0), np.std(data, 0)
    data = (data - data_mean) / np.clip(data_std, 1e-12, None)
    n_total=n_data_to_be_valued + n_val + n_test

    if len(data) >  n_total:
        X=data[:n_data_to_be_valued]
        y=target[:n_data_to_be_valued]
        X_val=data[n_data_to_be_valued:(n_data_to_be_valued+n_val)]
        y_val=target[n_data_to_be_valued:(n_data_to_be_valued+n_val)]
        X_test=data[(n_data_to_be_valued+n_val):(n_data_to_be_valued+n_val+n_test)]
        y_test=target[(n_data_to_be_valued+n_val):(n_data_to_be_valued+n_val+n_test)]
    else:
        assert False, f"Original dataset is less than n_data_to_be_valued + n_val + n_test. {len(data)} vs {n_total}. Try again with a smaller number for validation or test."

    print(f'Train X: {X.shape}')
    print(f'Val X: {X_val.shape}') 
    print(f'Test X: {X_test.shape}') 
    print('-'*30)
    
    return (X, y), (X_val, y_val), (X_test, y_test)
