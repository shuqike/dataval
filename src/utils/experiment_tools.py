import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import torchvision
from src.utils.data_tools import CustomDataset, MNIST_truncated


'''
noisy detection task
'''

def compute_f1_score_by_set(list_a, list_b):
    '''
    Comput F1 score for noisy detection task
    list_a : true flipped data points
    list_b : predicted flipped data points
    '''
    n_a, n_b=len(list_a), len(list_b)
    
    # among A, how many B's are selected
    n_intersection=len(set(list_b).intersection(list_a))
    recall=n_intersection/(n_a+1e-16)
    # among B, how many A's are selected
    precision=n_intersection/(n_b+1e-16)
    
    if recall > 0 and precision > 0:
        f1_score=1/((1/recall + 1/precision)/2)
    else:
        f1_score=0.
    return f1_score

def noisy_detection_core(value, noisy_index):
    # without kmeans algorithm (but requires prior knowledge of the number of noise labels)
    index_of_small_values=np.argsort(value)[:len(noisy_index)]
    recall=len([ind for ind in index_of_small_values if ind in noisy_index])/len(noisy_index)

    kmeans=KMeans(n_clusters=2, random_state=0).fit(value.reshape(-1, 1))
    
    # using kmeans label
    guess_index=np.where(kmeans.labels_ == np.argmin(kmeans.cluster_centers_))[0]
    f1_kmeans_label=compute_f1_score_by_set(noisy_index, guess_index)

    return [recall, f1_kmeans_label] 

def noisy_detection_experiment(value_dict, noisy_index):
    noisy_score_dict=dict()
    for key in value_dict.keys():
        noisy_score_dict[key]=noisy_detection_core(value_dict[key], noisy_index)

    noisy_dict={'Meta_Data': ['Recall', 'Kmeans_label'],
                'Results': noisy_score_dict}
    return noisy_dict

def infer_cluster_labels(kmeans, ground_truth_labels):
    inferred_labels = {}
    for i in range(kmeans.n_clusters):
        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)
        # append actual labels for each point in cluster
        labels.append(ground_truth_labels[index])
        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))
        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]
    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels

def create_noisy_mnist(noise_level='normal'):
    # prepare data
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated('../data', train=True, download=True, transform=transform, y_train='nmsl')
    mnist_test_ds = MNIST_truncated('../data', train=False, download=True, transform=transform, y_train='nmsl')

    x_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    x_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    x_train = x_train.data.numpy()
    y_train = y_train.data.numpy()
    x_test = x_test.data.numpy()
    y_test = y_test.data.numpy()

    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    # normalize the data to 0 - 1
    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.

    if noise_level == 'normal':
        n_clusters = 20
    elif noise_level == 'low':
        n_clusters = 64
    elif noise_level == 'high':
        n_clusters = 12
    elif noise_level == 'none':
        return x_train, y_train, x_test, y_test, []
    else:
        raise NotImplementedError('Please choose noise level from [normal, low, high, none]')
    # use kmeans as a noisy annotator
    kmeans = KMeans(n_clusters = 20)
    kmeans.fit(x_train)
    cluster_labels = infer_cluster_labels(kmeans, y_train)
    X_clusters = kmeans.predict(x_train)
    predicted_labels = infer_data_labels(X_clusters, cluster_labels)

    # find out the noisy ones in advance
    noisy_idxs = np.where(predicted_labels != y_train)[0]

    # replace training labels with noisy labels
    # then return data and noisy indices
    return x_train, predicted_labels, x_test, y_test, noisy_idxs

def online_noisy_detection_experiment(T=10, noise_level='normal'):
    """
    Args:
        T: time horizon
    """
    x_train, y_train, x_test, y_test, noisy_idxs = create_noisy_mnist(noise_level=noise_level)
    for t in range(T):
        pass

'''
point removal task
'''

def point_removal_experiment(value_dict, X, y, X_test, y_test, problem='clf'):
    removal_ascending_dict, removal_descending_dict=dict(), dict()
    for key in value_dict.keys():
        removal_ascending_dict[key]=point_removal_core(X, y, X_test, y_test, value_dict[key], ascending=True, problem=problem)
        removal_descending_dict[key]=point_removal_core(X, y, X_test, y_test, value_dict[key], ascending=False, problem=problem)
    random_array=point_removal_core(X, y, X_test, y_test, 'Random', problem=problem)
    removal_ascending_dict['Random']=random_array
    removal_descending_dict['Random']=random_array
    return {'ascending':removal_ascending_dict, 'descending':removal_descending_dict}

def point_removal_core(X, y, X_test, y_test, value_list, ascending=True, problem='clf'):
    n_sample=len(X)
    if value_list == 'Random':
        sorted_value_list=np.random.permutation(n_sample) 
    else:
        if ascending is True:
            sorted_value_list=np.argsort(value_list) # ascending order. low to high.
        else:
            sorted_value_list=np.argsort(value_list)[::-1] # descending order. high to low.
    
    accuracy_list=[]
    n_period = min(n_sample//200, 5) # we add 0.5% at each time
    for percentile in range(0, n_sample, n_period):
        '''
        We repeatedly remove 5% of entire data points at each step.
        The data points whose value belongs to the lowest group are removed first.
        The larger, the better
        '''
        sorted_value_list_tmp=sorted_value_list[percentile:]
        if problem == 'clf':
            try:
                clf=LogisticRegression() 
                clf.fit(X[sorted_value_list_tmp], y[sorted_value_list_tmp])
                model_score=clf.score(X_test, y_test)
            except:
                # if y[sorted_value_list_tmp] only has one class
                model_score=np.mean(np.mean(y[sorted_value_list_tmp])==y_test)
        else:
            try:
                model=LinearRegression() 
                model.fit(X[sorted_value_list_tmp], y[sorted_value_list_tmp])
                model_score=-np.mean(((y_test - model.predict(X_test))**2).reshape(-1)) # model.score(X_test, y_test)
            except:
                # if y[sorted_value_list_tmp] only has one class
                model_score=0

        accuracy_list.append(model_score)

    return accuracy_list
