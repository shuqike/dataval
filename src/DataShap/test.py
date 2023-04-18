import numpy as np

def generate_features(latent, dependency):
    features = []
    n = latent.shape[0]
    print('n', n)
    exp = latent
    holder = latent
    for order in range(1,dependency+1):
        features.append(np.reshape(holder,[n,-1]))
        print('features', features)
        exp = np.expand_dims(exp,-1)
        print('exp', exp)
        holder = exp * np.expand_dims(holder,1)
        print('holder', holder)
    return np.concatenate(features,axis=-1)

print(generate_features(np.array([1,2,3]), 2))