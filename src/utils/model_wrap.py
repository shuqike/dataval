import models.pt_cnn
from sklearn.linear_model import LogisticRegression


def return_model(model_family='logistic', **kwargs):
    if model_family == 'logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, 
                                   max_iter=max_iter, random_state=666)
    elif model_family == 'vit':
        pass
    return model
