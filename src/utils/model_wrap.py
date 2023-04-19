import models
from sklearn.linear_model import LogisticRegression


def return_model(model_family='logistic', **kwargs):
    if model_family == 'logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, 
                                   max_iter=max_iter, random_state=666)
    elif model_family == 'vit':
        pretrained = kwargs.get('pretrained', False)
        model = models.ViTbp16(pretrained)
    elif model_family == 'swin-tiny':
        pretrained = kwargs.get('pretrained', False)
        model = models.SwinTiny(pretrained)
    elif model_family == 'mobilenet':
        pretrained = kwargs.get('pretrained', False)
        model = models.MobileNet(pretrained)
    elif model_family == 'resnet-18':
        pretrained = kwargs.get('pretrained', False)
        model = models.ResNet18(pretrained)
    elif model_family == 'resnet-50':
        pretrained = kwargs.get('pretrained', False)
        model = models.ResNet50(pretrained)
    elif model_family == 'convnext-tiny':
        pretrained = kwargs.get('pretrained', False)
        model = models.ConvNeXTTiny(pretrained)
    return model
