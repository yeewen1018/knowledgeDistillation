import utils
import torch 
import torch.optim as optim
import torch.nn as nn 

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import itertools

import scipy
import scipy.cluster.hierarchy as sch


def get_generalist_model_cifar100(model_name, pretrained_model_path, trainloader, testloader): 
    """ Returns generalist model trained on the CIFAR-100 dataset. """

    generalist_model_name = 'cifar100_' + model_name
    generalist_model = torch.hub.load("chenyaofo/pytorch-cifar-models", generalist_model_name, pretrained = True)

    if pretrained_model_path is not None: 
        print("Using pre-trained generalist model")
        if torch.cuda.is_available(): 
            state_dict = torch.load(pretrained_model_path)
            generalist_model = generalist_model.cuda() 
        else: 
            state_dict = torch.load(pretrained_model_path, map_location = torch.device('cpu'))
        generalist_model.load_state_dict(state_dict)
        
    else: 
        if torch.cuda.is_available(): 
            generalist_model = generalist_model.cuda() 
        optimizer = optim.SGD(generalist_model.parameters(), lr = 0.001, momentum = 0.9, nesterov = True, weight_decay = 5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 80, gamma = 0.1)
        criterion = nn.CrossEntropyLoss()
        num_epochs = 10 
        utils.train_and_evaluate_scratch(trainloader, testloader, generalist_model, optimizer, scheduler, criterion, num_epochs, 'specialists/generalist_model_cifar100.pth')

    return generalist_model 


def get_confusable_classes(model, testloader): 
    """ Returns the clusters of confusable classes of a model trained on a set of data. """
    cf_matrix = utils.get_confusion_matrix(model, testloader)
    cf_matrix = pd.DataFrame(cf_matrix)

    if torch.cuda.is_available(): 
        X = cf_matrix.corr().values()
    else: 
        X = cf_matrix.corr() 

    d = sch.distance.pdist(X)
    L = sch.linkage(d, method = "complete")
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    columns = [cf_matrix.columns.tolist()[i] for i in list((np.argsort(ind)))]
    cf_matrix = cf_matrix.reindex(columns, axis = 1)

    utils.plot_corr(cf_matrix, size = 18)

    # There should be a way to automate this process of identifying clusters of confusable classes. 
    # But for now, we manually identify these clusters from the correlation plot. The generalist model 
    # used here is resnet20. The clusters of confusable classes for this model are written as below:   
    clusters = [[35, 46, 11, 98, 2], [55, 72, 3], [47, 52, 96, 59], [74, 50, 4, 63], [15, 19, 43], [64, 97, 36, 65], [65, 80, 34, 38], 
          [74, 50, 4, 63, 15, 19, 43, 64, 97, 36, 65, 65, 80, 34, 38], [95, 30, 73], [32, 67, 93], [95, 30, 73, 32, 67, 93],
           [74, 50, 4, 63, 15, 19, 43, 64, 97, 36, 65, 65, 80, 34, 38, 7, 24, 79, 26, 45, 6, 14, 78, 99, 18, 44, 27, 95, 30, 73, 32, 67, 93],
           [28, 40, 10, 61], [10, 61, 22], [28, 40, 10, 61, 22], [92, 70, 62, 54], [81, 13, 90]]

    return clusters 
