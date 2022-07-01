import torch
import torch.optim as optim
import torch.nn as nn

import utils
from kmeans_pytorch import kmeans, kmeans_predict 

def get_generalist_model(opt, trainloader, testloader): 

    generalist_model_name = 'cifar100_' + opt.model_type
    generalist_model = torch.hub.load("chenyaofo/pytorch-cifar-models", generalist_model_name, pretrained = True)

    if opt.pretrained_generalist_path is not None: 
        print("Using pre-trained generalist model")
        generalist_state_dict = torch.load(opt.pretrained_generalist_path, map_location = torch.device('cpu'))
        generalist_model.load_state_dict(generalist_state_dict)
        if torch.cuda.is_available(): 
            generalist_model = generalist_model.cuda()
    else: 
        print("No pretrained path available. Training a new model as the generalist")
        optimizer = optim.SGD(generalist_model.parameters(), lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = opt.lr_step_size, gamma = opt.lr_scheduler_gamma)
        criterion = nn.CrossEntropyLoss() 
        model_path = opt.pretrained_generalist_path
        utils.train_and_evaluate_scratch(trainloader, testloader, generalist_model, optimizer, scheduler, criterion, opt.generalist_num_train_epochs, model_path)
    
    return generalist_model 


def get_specialist_subsets(opt, generalist_model, testloader):
    """
    Apply k-means clustering on covariance matrix of generalist model 
    to obtain subsets of specialists. 
    """
    
    if opt.predefined_specialist_subsets:  
        sub_classes = [[12, 13, 17, 23, 33, 37, 47, 49, 52, 56, 58, 59, 60, 68, 69, 71, 76, 81, 85, 89, 90, 96], 
               [0, 1, 2, 6, 7, 14, 18, 24, 26, 36, 44, 45, 51, 53, 54, 57, 62, 70, 77, 78, 79, 82, 83, 92, 99], 
               [5, 8, 9, 10, 11, 16, 20, 22, 25, 28, 35, 39, 40, 41, 46, 48, 61, 84, 86, 87, 94, 98], 
               [3, 4, 15, 19, 21, 29, 31, 34, 38, 42, 43, 50, 55, 63, 64, 65, 66, 74, 75, 80, 88, 97], 
               [27, 30, 32, 67, 72, 73, 91, 93, 95], 
                [12, 17, 23, 33, 37, 47, 49, 52, 56, 59, 60, 68, 69, 71, 76, 81, 85, 90, 96],
               [3, 4, 21, 29, 31, 34, 38, 42, 43, 63, 64, 66, 74, 80, 88, 97],
               [27, 30, 55, 72, 95],
               [6, 7, 14, 26, 44, 50, 51, 77, 78, 79, 93],
               [1, 18, 24, 32, 45, 67, 73, 91, 99],
               [2, 11, 15, 19, 35, 36, 46, 65, 75, 98]]
        return sub_classes 

    outputs_array = torch.zeros(10000, opt.num_classes)
    generalist_model.eval() 
    i = 0 
    with torch.no_grad(): 
        for data in testloader: 
            inputs, labels = data
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda() 

            outputs = generalist_model(inputs)
            num_examples_per_batch = labels.shape[0]
            for index in range(num_examples_per_batch): 
                outputs_array[i] = outputs[index].detach().cpu()
                i += 1
    
    covariance_matrix = torch.cov(outputs_array.T)

    # K-means clustering 
    cluster_ids_x, cluster_centers = kmeans(X = covariance_matrix, num_clusters = opt.num_specialists, distance = 'euclidan', 
                                            device = torch.device('cuda:0') if torch.cuda.is_availale() else torch.device('cpu'))

    sub_classes = [] 
    for i in range(opt.num_specialists): 
        sub_class = [] 
        for j in range(len(cluster_ids_x)): 
            if cluster_ids_x[j] == i: 
                sub_class.append(j)
        sub_classes.append(sub_class)
    return sub_classes

