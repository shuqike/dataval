import pickle
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.frameworks.valuator import StaticValuator
from src.frameworks.outofbag import DataOOB


class SistValuator:
    def __init__(self, X, y, X_val, y_val, problem, dargs):
        """
        Args:
            (X, y): (inputs, outputs) to be valued.
            (X_val, y_val): (inputs, outputs) to be used for utility evaluation.
            problem: "clf" 
            dargs: arguments of the experimental setting
        """
        self.X=X
        self.y=y
        self.X_val=X_val
        self.y_val=y_val
        self.problem=problem
        self.dargs=dargs
        self._initialize_instance()

    def _initialize_instance(self):
        # intialize dictionaries
        self.run_id=self.dargs['run_id']
        self.n_tr=self.dargs['n_data_to_be_valued']
        self.n_val=self.dargs['n_val']
        self.n_trees=self.dargs['n_trees']
        self.is_noisy=self.dargs['is_noisy']
        self.model_family=self.dargs['model_family']
        self.model_name=f'{self.n_tr}:{self.n_val}:{self.n_trees}:{self.is_noisy}'

        # set random seed
        np.random.seed(self.run_id)

        # create placeholders
        self.data_value_dict=defaultdict(list)
        self.time_dict=defaultdict(list)
        self.noisy_detect_dict=defaultdict(list)
        self.removal_dict=defaultdict(list)

    def evaluate_baseline_models(self, X_test, y_test):
        if self.problem == 'clf':
            model_logistic=LogisticRegression()
            model_logistic.fit(self.X, self.y)
            acc_logistic=model_logistic.score(X_test, y_test)

            model_rf=RandomForestClassifier()
            model_rf.fit(self.X, self.y)
            acc_rf=model_rf.score(X_test, y_test)
            acc_tree_1=np.mean([model_tmp.score(X_test, y_test) for model_tmp in model_rf.estimators_])

            model_tree=DecisionTreeClassifier()
            model_tree.fit(self.X, self.y)
            acc_tree=model_tree.score(X_test, y_test)

            model_knn=KNeighborsClassifier(n_neighbors=10)
            model_knn.fit(self.X, self.y)
            acc_knn=model_knn.score(X_test, y_test)

            print(f'Logistic {acc_logistic:.3f}')
            print(f'RF {acc_rf:.3f}')
            print(f'Avg tree {acc_tree_1:.3f}')
            print(f'Tree {acc_tree:.3f}')
            print(f'KNN {acc_knn:.3f}')
            self.baseline_score_dict={'Meta_Data': ['Logistic', 'RF', 'Avg_tree', 'Tree', 'KNN'],
                                      'Results': [acc_logistic, acc_rf, acc_tree_1, acc_tree, acc_knn]}
        else:
            model_linear=LinearRegression()
            model_linear.fit(self.X, self.y)
            r2_linear=model_linear.score(X_test, y_test)

            model_rf=RandomForestRegressor()
            model_rf.fit(self.X, self.y)
            r2_rf=model_rf.score(X_test, y_test)
            r2_tree_1=np.mean([model_tmp.score(X_test, y_test) for model_tmp in model_rf.estimators_])

            model_tree=DecisionTreeRegressor()
            model_tree.fit(self.X, self.y)
            r2_tree=model_tree.score(X_test, y_test)

            model_knn=KNeighborsRegressor(n_neighbors=10)
            model_knn.fit(self.X, self.y)
            r2_knn=model_knn.score(X_test, y_test)

            print(f'Linear {r2_linear:.3f}')
            print(f'RF {r2_rf:.3f}')
            print(f'Avg tree {r2_tree_1:.3f}')
            print(f'Tree {r2_tree:.3f}')
            print(f'KNN {r2_knn:.3f}')
            self.baseline_score_dict={'Meta_Data': ['Linear', 'RF', 'Avg_tree', 'Tree', 'KNN'],
                                      'Results': [r2_linear, r2_rf, r2_tree_1, r2_tree, r2_knn]}

    def save_results(self, runpath, dataset, dargs_ind, noisy_index):
        self.sparsity_dict=defaultdict(list)
        for key in self.data_value_dict:
            self.sparsity_dict[key]=np.mean(self.data_value_dict[key]==0)

        print('-'*50)
        print('Save results')
        print('-'*50)
        result_dict={'data_value': self.data_value_dict,
                     'sparse': self.sparsity_dict,
                     'time': self.time_dict,
                     'noisy': self.noisy_detect_dict,
                     'removal': self.removal_dict,
                     'dargs':self.dargs,
                     'dataset':dataset,
                     'input_dim':self.X.shape[1],
                     'model_name':self.model_name,
                     'noisy_index':noisy_index,
                     'baseline_score':self.baseline_score_dict
        }
        with open(runpath+f'/run_id{self.run_id}_{dargs_ind}.pkl', 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done! path: {runpath}, run_id: {self.run_id}.',flush=True)