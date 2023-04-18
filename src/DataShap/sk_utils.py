'''Refer to https://github.com/amiratag/DataShapley/blob/master/shap_utils.py
- Upgraded some TensorFlow1 API to TensorFlow2 API
- Added some comments
'''
import numpy as np
import inspect
from scipy.stats import logistic
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, f1_score
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class ShapNN(object):
    
    def __init__(self, mode, hidden_units=[100], learning_rate=0.001, 
                 dropout = 0., activation=None, initializer=None,
                 weight_decay=0.0001, optimizer='adam', batch_size=128,
                 warm_start=False, max_epochs=100, validation_fraction=0.1,
                 early_stopping=0, address=None, test_batch_size=1000,
                 random_seed=666):
        
        self.mode = mode
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.hidden_units = hidden_units
        self.initializer = initializer
        self.activation = activation
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.address = address
        self._extra_train_ops = []
        self.random_seed = random_seed
        self.is_built = False

    def prediction_cost(self, X_test, y_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.test_batch_size
        assert len(set(y_test)) == self.num_classes, 'Number of classes does not match!'
        with self.graph.as_default():
            losses = []
            idxs = np.arange(len(X_test))            
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                losses.append(self.sess.run(self.prediction_loss, {self.input_ph:X_test[batch],
                                                                   self.labels:y_test[batch]}))
            return np.mean(losses)     
        
    def score(self, X_test, y_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.test_batch_size
        assert len(set(y_test)) == self.num_classes, 'Number of classes does not match!'
        with self.graph.as_default():
            scores = []
            idxs = np.arange(len(X_test))     
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                scores.append(self.sess.run(self.prediction_score, {self.input_ph:X_test[batch],
                                                                   self.labels:y_test[batch]}))
            return np.mean(scores)
        
    def predict_proba(self, X_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.test_batch_size
        with self.graph.as_default():
            probs = []
            idxs = np.arange(len(X_test))     
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                probs.append(self.sess.run(self.probs, {self.input_ph:X_test[batch]}))
            return np.concatenate(probs, axis=0)    
        
    def predict_log_proba(self, X_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.test_batch_size
        with self.graph.as_default():
            probs = []
            idxs = np.arange(len(X_test))            
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                probs.append(self.sess.run(self.probs, {self.input_ph:X_test[batch]}))
            return np.log(np.clip(np.concatenate(probs), 1e-12, None))   
        
    def cost(self, X_test, y_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.batch_size
        with self.graph.as_default():
            losss = []
            idxs = np.arange(len(X_test))            
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                losss.append(self.sess.run(self.prediction_loss, {self.input_ph:X_test[batch],
                                                                   self.labels:y_test[batch]}))
            return np.mean(losss)
    
    def predict(self, X_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.batch_size
        with self.graph.as_default():
            predictions = []
            idxs = np.arange(len(X_test))
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                predictions.append(self.sess.run(self.predictions, {self.input_ph:X_test[batch]}))
            return np.concatenate(predictions)
        
    def fit(self, X, y, X_val=None, y_val=None, sources=None, max_epochs=None,
            batch_size=None, save=False, load=False, sample_weight=None,
            metric='accuracy'):
        
        self.num_classes = len(set(y))
        self.metric = metric
        if max_epochs is None:
            max_epochs = self.max_epochs
        if batch_size is None:
            batch_size = self.batch_size
        if not self.is_built:
            self.graph = tf.Graph() 
            with self.graph.as_default():
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth=True
                self.sess = tf.compat.v1.Session(config=config)
        with self.graph.as_default():
            tf.random.set_seed(self.random_seed)
            try:
                self.global_step = tf.compat.v1.train.create_global_step()
            except ValueError:
                self.global_step = tf.compat.v1.train.get_global_step()
            if not self.is_built:
                self._build_model(X, y)
                self.saver = tf.compat.v1.train.Saver()
            self._initialize()
            if len(X):
                if X_val is None and self.validation_fraction * len(X) > 2:
                    X_train, X_val, y_train, y_val, sample_weight, _ = train_test_split(
                        X, y, sample_weight, test_size=self.validation_fraction)
                else:
                    X_train, y_train = X, y
                self._train_model(X_train, y_train, X_val=X_val, y_val=y_val,
                                  max_epochs=max_epochs, batch_size=batch_size,
                                  sources=sources, sample_weight=sample_weight)
                if save and self.address is not None:
                    self.saver.save(self.sess, self.address)

    def _train_model(self, X, y, X_val, y_val, max_epochs, batch_size, 
                     sources=None, sample_weight=None):
        
        
        assert len(X)==len(y), 'Input and labels not the same size'
        self.history = {'metrics':[], 'idxs':[]}
        stop_counter = 0
        best_performance = None
        for epoch in range(max_epochs):
            vals_metrics, idxs = self._one_epoch(
                X, y, X_val, y_val, batch_size, sources=sources, sample_weight=sample_weight)
            self.history['idxs'].append(idxs)
            self.history['metrics'].append(vals_metrics)
            if self.early_stopping and X_val is not None:
                current_performance = np.mean(val_acc)
                if best_performance is None:
                    best_performance = current_performance
                if current_performance > best_performance:
                    best_performance = current_performance
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter > self.early_stopping:
                        break

    def _one_epoch(self, X, y, X_val, y_val, batch_size, sources=None, sample_weight=None):
        
        vals = []
        if sources is None:
            if sample_weight is None:
                idxs = np.random.permutation(len(X))
            else:
                idxs = np.random.choice(len(X), len(X), p=sample_weight/np.sum(sample_weight))    
            batches = [idxs[k*batch_size:(k+1) * batch_size]
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            idxs = batches
        else:
            idxs = np.random.permutation(len(sources.keys()))
            batches = [sources[i] for i in idxs]
        for batch_counter, batch in enumerate(batches):
            self.sess.run(self.train_op, 
                          {self.input_ph:X[batch], self.labels:y[batch],
                           self.dropout_ph:self.dropout})
            if X_val is not None:
                if self.metric=='accuracy':
                    vals.append(self.score(X_val, y_val))
                elif self.metric=='f1':
                    vals.append(f1_score(y_val, self.predict(X_val)))
                elif self.metric=='auc':
                    vals.append(roc_auc_score(y_val, self.predict_proba(X_val)[:,1]))
                elif self.metric=='xe':
                    vals.append(-self.prediction_cost(X_val, y_val))
        return np.array(vals), np.array(idxs)
    
    def _initialize(self):
        
        uninitialized_vars = []
        if self.warm_start:
            for var in tf.compat.v1.global_variables():
                try:
                    self.sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
        else:
            uninitialized_vars = tf.compat.v1.global_variables()
        self.sess.run(tf.compat.v1.variables_initializer(uninitialized_vars))
        
    def _build_model(self, X, y):
        
        self.num_classes = len(set(y))
        if self.initializer is None:
            initializer = tf.keras.initializers.VarianceScaling(distribution='uniform')
        if self.activation is None:
            activation = lambda x: tf.nn.relu(x)
        self.input_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,) + X.shape[1:], name='input')
        self.dropout_ph = tf.compat.v1.placeholder_with_default(
            tf.constant(0., dtype=tf.float32), shape=(), name='dropout')
        if self.mode=='regression':
            self.labels = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, ), name='label')
        else:
            self.labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, ), name='label')
        x = tf.reshape(self.input_ph, shape=(-1, np.prod(X.shape[1:])))
        for layer, hidden_unit in enumerate(self.hidden_units):
            with tf.compat.v1.variable_scope('dense_{}'.format(layer)):
                x = self._dense(x, hidden_unit, dropout=self.dropout_ph, 
                           initializer=self.initializer, activation=activation)
        with tf.compat.v1.variable_scope('final'):
            self.prelogits = x
            self._final_layer(self.prelogits, self.num_classes, self.mode)
        self._build_train_op()
        
    def _build_train_op(self):
        
        """Build taining specific ops for the graph."""
        learning_rate = tf.constant(self.learning_rate, tf.float32) ##fixit
        trainable_variables = tf.compat.v1.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        self.grad_flat = tf.concat([tf.reshape(grad, (-1, 1)) for grad in grads], axis=0)
        if self.optimizer == 'sgd':
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer == 'mom':
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, 0.9)
        elif self.optimizer == 'adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')
        train_ops = [apply_op] + self._extra_train_ops + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        previous_ops = [tf.group(*train_ops)]
        with tf.control_dependencies(previous_ops):
            self.train_op = tf.no_op(name='train')   
        self.is_built = True
    
    def _final_layer(self, x, num_classes, mode):
        
        if mode=='regression':
            self.logits = self._dense(x, 1, dropout=self.dropout_ph)
            self.predictions = tf.math.reduce_sum(self.logits, axis=-1)
            regression_loss = tf.nn.l2_loss(self.predictions - self.labels) ##FIXIT
            self.prediction_loss = tf.math.reduce_mean(regression_loss, name='l2')
            residuals = self.predictions - self.labels
            var_predicted = tf.math.reduce_mean(residuals**2) - tf.math.reduce_mean(residuals)**2
            var_labels = tf.math.reduce_mean(self.labels**2) - tf.math.reduce_mean(self.labels)**2
            self.prediction_score = 1 - var_predicted/(var_labels + 1e-12)
        else:
            self.logits = self._dense(x, num_classes, dropout=self.dropout_ph)
            self.probs = tf.nn.softmax(self.logits)
            xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=tf.cast(self.labels, tf.int32))
            self.prediction_loss = tf.math.reduce_mean(xent_loss, name='xent')
            self.predictions = tf.math.argmax(self.probs, axis=-1, output_type=tf.int32)
            correct_predictions = tf.math.equal(self.predictions, self.labels)
            self.prediction_score = tf.math.reduce_mean(tf.cast(correct_predictions, tf.float32))
        self.loss = self.prediction_loss + self._reg_loss()
                
    def _dense(self, x, out_dim, dropout=tf.constant(0.), initializer=None, activation=None):
        
        if initializer is None:
            initializer = tf.keras.initializers.VarianceScaling(distribution='uniform')
        w = tf.compat.v1.get_variable('DW', [x.get_shape()[1], out_dim], initializer=initializer)
        b = tf.compat.v1.get_variable('Db', [out_dim], initializer=tf.constant_initializer())
        x = tf.nn.dropout(x, 1. - dropout)
        if activation:
            x = activation(x)
        return tf.compat.v1.nn.xw_plus_b(x, w, b)
    
    def _reg_loss(self, order=2):
        """Regularization loss for weight decay."""
        losss = []
        for var in tf.compat.v1.trainable_variables():
            if var.op.name.find(r'DW') > 0 or var.op.name.find(r'CW') > 0: ##FIXIT
                if order==2:
                    losss.append(tf.nn.l2_loss(var))
                elif order==1:
                    losss.append(tf.math.abs(var))
                else:
                    raise ValueError("Invalid regularization order!")
        return tf.math.multiply(self.weight_decay, tf.math.add_n(losss))


class CShapNN(ShapNN):
    
    def __init__(self, mode, hidden_units=[100], kernel_sizes=[], 
                 strides=None, channels=[], learning_rate=0.001, 
                 dropout = 0., activation=None, initializer=None, global_averaging=False,
                weight_decay=0.0001, optimizer='adam', batch_size=128, 
                warm_start=False, max_epochs=100, validation_fraction=0.1,
                early_stopping=0, address=None, test_batch_size=1000, random_seed=666):
        
        self.mode = mode
        self.test_batch_size = test_batch_size
        self.kernels = []#FIXIT
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.global_averaging = global_averaging
        assert len(channels)==len(kernel_sizes), 'Invalid channels or kernel_sizes'
        if strides is None:
            self.strides = [1] * len(kernel_sizes)
        else:
            self.strides = strides
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.initializer = initializer
        self.activation = activation
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.address = address
        self._extra_train_ops = []
        self.random_seed = random_seed
        self.graph = tf.Graph()
        self.is_built = False
        with self.graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.compat.v1.Session(config=config)
            
    def _conv(self, x, filter_size, out_filters, strides, activation=None):
        
        in_filters = int(x.get_shape()[-1])
        n = filter_size * filter_size * out_filters
        kernel = tf.compat.v1.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/n)))
        self.kernels.append(kernel)
        x = tf.nn.conv2d(x, kernel, strides, padding='SAME')
        if activation:
            x = activation(x)
        return x
    
    def _stride_arr(self, stride):
        
        if isinstance(stride, int):
            return [1, stride, stride, 1]
        if len(stride)==2:
            return [1, stride[0], stride[1], 1]
        if len(stride)==4:
            return stride
        raise ValueError('Invalid value!')  
        
    def _build_model(self, X, y):
        
        
        if self.initializer is None:
            initializer = tf.keras.initializers.VarianceScaling(distribution='uniform')
        if self.activation is None:
            activation = lambda x: tf.nn.relu(x)
        self.input_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,) + X.shape[1:], name='input')
        self.dropout_ph = tf.compat.v1.placeholder_with_default(
            tf.constant(0., dtype=tf.float32), shape=(), name='dropout')
        if self.mode=='regression':
            self.labels = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, ), name='label')
        else:
            self.labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, ), name='label')
        if len(X.shape[1:]) == 2:
            x = tf.reshape(self.input_ph, [-1, X.shape[0], X.shape[1], 1])
        else:
            x = self.input_ph
        for layer, (kernel_size, channels, stride) in enumerate(zip(
            self.kernel_sizes, self.channels, self.strides)):
            with tf.compat.v1.variable_scope('conv_{}'.format(layer)):
                x = self._conv(x, kernel_size, channels, self._stride_arr(stride), activation=activation)
        if self.global_averaging:
            x = tf.math.reduce_mean(x, axis=(1,2))
        else:
            x = tf.reshape(x, shape=(-1, np.prod(x.get_shape()[1:])))
        for layer, hidden_unit in enumerate(self.hidden_units):
            with tf.compat.v1.variable_scope('dense_{}'.format(layer)):
                x = self._dense(x, hidden_unit, dropout=self.dropout_ph, 
                           initializer=self.initializer, activation=activation)
                
        with tf.compat.v1.variable_scope('final'):
            self.prelogits = x
            self._final_layer(self.prelogits, len(set(y)), self.mode)
        self._build_train_op()


def convergence_plots(marginals):
    '''Plot the cumulative sum of the elements along a given axis.
    See notebooks/dataShap.ipynb
    '''
    plt.rcParams['figure.figsize'] = 15,15
    for i, idx in enumerate(np.arange(min(25, marginals.shape[-1]))):
        plt.subplot(5,5,i+1)
        plt.plot(np.cumsum(marginals[:, idx])/np.arange(1, len(marginals)+1))

def is_integer(array):
    '''Check if the array elements are all integers
    Remark: we probably won't use this function throughout this project
    '''
    return (np.equal(np.mod(array, 1), 0).mean()==1)

def is_fitted(model):
    '''Checks if model object has any attributes ending with an underscore
    Refer to Python PEP 8:
    _single_leading_underscore: weak "internal use" indicator. E.g. from M import * does not import objects whose name starts with an underscore.
    __double_leading_underscore: when naming a class attribute, invokes name mangling (inside class FooBar, __boo becomes _FooBar__boo; see below).
    __double_leading_and_trailing_underscore__: "magic" objects or attributes that live in user-controlled namespaces. E.g. __init__, __import__ or __file__. Never invent such names; only use them as documented.
    Remark: we probably won't use this function throughout this project
    '''
    return 0 < len( [k for k,v in inspect.getmembers(model) if k.endswith('_') and not k.startswith('__')] )

def return_model(mode, **kwargs):
    '''Might be problematic because the following code uses TensorFlow 1 instead of TensorFlow 2
    '''
    if inspect.isclass(mode):
        assert getattr(mode, 'fit', None) is not None, 'Custom model family should have a fit() method'
        model = mode(**kwargs)
    elif mode=='logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, 
                                 max_iter=max_iter, random_state=666)
    elif mode=='Tree':
        model = DecisionTreeClassifier(random_state=666)
    elif mode=='RandomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='GB':
        n_estimators = kwargs.get('n_estimators', 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='AdaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='SVC':
        kernel = kwargs.get('kernel', 'rbf')
        model = SVC(kernel=kernel, random_state=666)
    elif mode=='LinearSVC':
        model = LinearSVC(loss='hinge', random_state=666)
    elif mode=='GP':
        model = GaussianProcessClassifier(random_state=666)
    elif mode=='KNN':
        n_neighbors = kwargs.get('n_neighbors', 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif mode=='NB':
        model = MultinomialNB()
    elif mode=='linear':
        model = LinearRegression(random_state=666)
    elif mode=='ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=666)
    elif 'conv' in mode:
        tf.compat.v1.reset_default_graph()
        address = kwargs.get('address', 'weights/conv')
        hidden_units = kwargs.get('hidden_layer_sizes', [20])
        activation = kwargs.get('activation', 'relu')
        weight_decay = kwargs.get('weight_decay', 1e-4)
        learning_rate = kwargs.get('learning_rate', 0.001)
        max_iter = kwargs.get('max_iter', 1000)
        early_stopping= kwargs.get('early_stopping', 10)
        warm_start = kwargs.get('warm_start', False)
        batch_size = kwargs.get('batch_size', 256)
        kernel_sizes = kwargs.get('kernel_sizes', [5])
        strides = kwargs.get('strides', [5])
        channels = kwargs.get('channels', [1])
        validation_fraction = kwargs.get('validation_fraction', 0.)
        global_averaging = kwargs.get('global_averaging', 0.)
        optimizer = kwargs.get('optimizer', 'sgd')
        if mode=='conv':
            model = CShapNN(mode='classification', batch_size=batch_size, max_epochs=max_iter,
                          learning_rate=learning_rate, 
                          weight_decay=weight_decay, validation_fraction=validation_fraction,
                          early_stopping=early_stopping,
                         optimizer=optimizer, warm_start=warm_start, address=address,
                          hidden_units=hidden_units,
                          strides=strides, global_averaging=global_averaging,
                         kernel_sizes=kernel_sizes, channels=channels, random_seed=666)
        elif mode=='conv_reg':
            model = CShapNN(mode='regression', batch_size=batch_size, max_epochs=max_iter,
                          learning_rate=learning_rate, 
                          weight_decay=weight_decay, validation_fraction=validation_fraction,
                          early_stopping=early_stopping,
                         optimizer=optimizer, warm_start=warm_start, address=address,
                          hidden_units=hidden_units,
                          strides=strides, global_averaging=global_averaging,
                         kernel_sizes=kernel_sizes, channels=channels, random_seed=666)
    elif 'NN' in mode:
        solver = kwargs.get('solver', 'adam')
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (20,))
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = list(hidden_layer_sizes)
        activation = kwargs.get('activation', 'relu')
        learning_rate_init = kwargs.get('learning_rate', 0.001)
        max_iter = kwargs.get('max_iter', 5000)
        early_stopping= kwargs.get('early_stopping', False)
        warm_start = kwargs.get('warm_start', False)
        if mode=='NN':
            model = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation, learning_rate_init=learning_rate_init,
                                warm_start = warm_start, max_iter=max_iter,
                                early_stopping=early_stopping)
        if mode=='NN_reg':
            model = MLPRegressor(solver=solver, hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation, learning_rate_init=learning_rate_init,
                                warm_start = warm_start, max_iter=max_iter, early_stopping=early_stopping)
    else:
        raise ValueError("Invalid mode!")
    return model

def generate_features(latent, dependency):
    features = []
    n = latent.shape[0]
    exp = latent
    holder = latent
    for order in range(1,dependency+1):
        features.append(np.reshape(holder,[n,-1]))
        exp = np.expand_dims(exp,-1)
        holder = exp * np.expand_dims(holder,1)
    return np.concatenate(features,axis=-1)

def label_generator(problem, X, param, difficulty=1, beta=None, important=None):
    if important is None or important > X.shape[-1]:
        important = X.shape[-1]
    dim_latent = sum([important**i for i in range(1, difficulty+1)])
    if beta is None:
        beta = np.random.normal(size=[1, dim_latent])
    important_dims = np.random.choice(X.shape[-1], important, replace=False)
    funct_init = lambda inp: np.sum(beta * generate_features(inp[:,important_dims], difficulty), -1)
    batch_size = max(100, min(len(X), 10000000//dim_latent))
    y_true = np.zeros(len(X))
    while True:
        try:
            for itr in range(int(np.ceil(len(X)/batch_size))):
                y_true[itr * batch_size: (itr+1) * batch_size] = funct_init(
                    X[itr * batch_size: (itr+1) * batch_size])
            break
        except MemoryError:
            batch_size = batch_size//2
    mean, std = np.mean(y_true), np.std(y_true)
    funct = lambda x: (np.sum(beta * generate_features(
        x[:, important_dims], difficulty), -1) - mean) / std
    y_true = (y_true - mean)/std
    if problem == 'classification':
        y_true = logistic.cdf(param * y_true)
        y = (np.random.random(X.shape[0]) < y_true).astype(int)
    elif problem == 'regression':
        y = y_true + param * np.random.normal(size=len(y_true))
    else:
        raise ValueError('Invalid problem specified!')
    return beta, y, y_true, funct

def one_iteration(clf, X, y, X_test, y_test, mean_score, tol=0.0, c=None, metric='accuracy'):
    '''Runs one iteration of TMC-Shapley (Truncated Monte Carlo Shapley).
    clf: the classifier
    tol: performance tolerance
    c: the map from X to the real datum index
    '''
    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:,1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")
    if c is None:
        c = {i:np.array([i]) for i in range(len(X))}

    # idxs: Random permutation of train data points
    idxs, marginal_contribs = np.random.permutation(len(c.keys())), np.zeros(len(X))
    # Count frequency of occurrences of each integer label;
    # then set the maximum frequency as a baseline score;
    # which is the best result by blind guessing
    new_score = np.max(np.bincount(y)) * 1./len(y) if np.mean(y//1 == y/1)==1 else 0.
    # 'start' controls the starting datum
    start = 0
    if start:
        X_batch, y_batch =\
        np.concatenate([X[c[idx]] for idx in idxs[:start]]), np.concatenate([y[c[idx]] for idx in idxs[:start]])
    else:
        X_batch, y_batch = np.zeros((0,) +  tuple(X.shape[1:])), np.zeros(0).astype(int)
    # Start from the starting datum
    for n, idx in enumerate(idxs[start:]):
        try:
            clf = clone(clf)
        except:
            clf.fit(np.zeros((0,) +  X.shape[1:]), y)
        old_score = new_score
        # Add the new datum to the training batch
        X_batch, y_batch = np.concatenate([X_batch, X[c[idx]]]), np.concatenate([y_batch, y[c[idx]]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Train the model and get the current score
                clf.fit(X_batch, y_batch)
                temp_score = score_func(clf, X_test, y_test)
                if temp_score>-1 and temp_score<1.: #Removing measningless r2 scores
                    new_score = temp_score
            except:
                continue
        # Update the marginal contribution with the weighted score difference
        marginal_contribs[c[idx]] = (new_score - old_score)/len(c[idx])
        # if the score up is smaller than the performance tolerance,
        # i.e. cannot improve much with this training proceedure, then stop
        if np.abs(new_score - mean_score)/mean_score < tol:
            break
    return marginal_contribs, idxs

def marginals(clf, X, y, X_test, y_test, c=None, tol=0., trials=3000, mean_score=None, metric='accuracy'):
    '''Runs the full TMC-Shapley (Truncated Monte Carlo Shapley) for so many trials
    clf: the classifier
    tol: performance tolerance
    c: the map from X to the real datum index
    '''
    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:,1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")
    if mean_score is None:
        accs = []
        for _ in range(100):
            bag_idxs = np.random.choice(len(y_test), len(y_test))
            accs.append(score_func(clf, X_test[bag_idxs], y_test[bag_idxs]))
        mean_score = np.mean(accs)
    marginals, idxs = [], []
    for trial in range(trials):
        if 10*(trial+1)/trials % 1 == 0:
            print('{} out of {}'.format(trial + 1, trials))
        marginal, idx = one_iteration(clf, X, y, X_test, y_test, mean_score, tol=tol, c=c, metric=metric)
        marginals.append(marginal)
        idxs.append(idx)
    return np.array(marginals), np.array(idxs)

def early_stopping(marginals, idxs, stopping):
    stopped_marginals = np.zeros_like(marginals)
    for i in range(len(marginals)):
        stopped_marginals[i][idxs[i][:stopping]] = marginals[i][idxs[i][:stopping]]
    return np.mean(stopped_marginals, 0)

def error(mem):
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)

def my_accuracy_score(clf, X, y):
    probs = clf.predict_proba(X)
    predictions = np.argmax(probs, -1)
    return np.mean(np.equal(predictions, y))

def my_f1_score(clf, X, y):
    predictions = clf.predict(X)
    if len(set(y)) == 2:
        return f1_score(y, predictions)
    return f1_score(y, predictions, average='macro')

def my_auc_score(clf, X, y):
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    return roc_auc_score(y, true_probs)

def my_xe_score(clf, X, y):
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    true_log_probs = np.log(np.clip(true_probs, 1e-12, None))
    return np.mean(true_log_probs)
