import inspect
import sklearn


def return_model(mode, **kwargs):
    if inspect.isclass(mode):
        assert getattr(mode, 'fit', None) is not None, 'Custom model family should have a fit() method'
        model = mode(**kwargs)
    elif mode=='logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        max_iter = kwargs.get('max_iter', 5000)
        model = sklearn.linear_model.LogisticRegression(
            solver=solver,
            n_jobs=n_jobs,
            max_iter=max_iter,
            random_state=666
        )
    elif mode=='decisionTree':
        model = sklearn.tree.DecisionTreeClassifier(random_state=666)
    elif mode=='randomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='gradBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='adaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = sklearn.ensembleAdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='SVC':
        kernel = kwargs.get('kernel', 'rbf')
        model = sklearn.svm.SVC(kernel=kernel, random_state=666)
    elif mode=='linearSVC':
        model = sklearn.svm.LinearSVC(loss='hinge', random_state=666)
    elif mode=='gaussianProc':
        model = sklearn.gaussian_process.GaussianProcessClassifier(random_state=666)
    elif mode=='KNN':
        n_neighbors = kwargs.get('n_neighbors', 5)
        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    elif mode=='NB':
        model = sklearn.naive_bayes.MultinomialNB()
    elif mode=='linear':
        model = sklearn.linear_model.LinearRegression(random_state=666)
    elif mode=='ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = sklearn.linear_model.Ridge(alpha=alpha, random_state=666)
    elif mode=='':
        pass
    else:
        raise ValueError("Invalid mode!")
    return model
