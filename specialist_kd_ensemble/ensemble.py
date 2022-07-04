import torch 
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 


def get_predicted_class(num_classes, baseline_class, generalist_output, specialist_outputs, sub_classes, device): 

    q_dist = torch.randn(num_classes, requires_grad = True)
    q_dist.to(device)

    optimizer = optim.Adam([q_dist], lr = 0.01)
    criterion = nn.KLDivLoss() 
    previous_loss = 10000.0
    loss = criterion(F.log_softmax(q_dist), F.softmax(generalist_output))
    
    num_specialists = len(sub_classes)
    for i in range(num_specialists):
        sub_class = sub_classes[i]
        if baseline_class in sub_class: 
            modified_q_dist = np.zeros(len(sub_class) + 1)
            q_dist_copy = q_dist.clone() 
            for index in range(len(sub_class)): 
                modified_q_dist[index] = q_dist_copy[sub_class[index]]
                q_dist_copy[sub_class[index]] = 0.0 
                
            modified_q_dist[len(sub_class)] = torch.sum(q_dist_copy)
            modified_q_dist = torch.tensor(modified_q_dist)
            modified_q_dist = modified_q_dist.to(device)
            loss += criterion(F.log_softmax(modified_q_dist), F.softmax(specialist_outputs[i]))
    
    #while loss > 1e-3: 
    while previous_loss > loss and loss > 1e-3: 
        previous_loss = loss 
        loss.backward(retain_graph = True)
        optimizer.step()
        optimizer.zero_grad()
        loss = criterion(F.log_softmax(q_dist), F.softmax(generalist_output))
        
        for i in range(num_specialists): 
            sub_class = sub_classes[i]
            if baseline_class in sub_class: 
                modified_q_dist = np.zeros(len(sub_class) + 1)
                q_dist_copy = q_dist.clone() 
                for index in range(len(sub_class)): 
                    modified_q_dist[index] = q_dist_copy[sub_class[index]]
                    q_dist_copy[sub_class[index]] = 0.0 
                modified_q_dist[len(sub_class)] = torch.sum(q_dist_copy)
                modified_q_dist = torch.tensor(modified_q_dist)
                modified_q_dist.to(device)
                loss += criterion(F.log_softmax(modified_q_dist), F.softmax(specialist_outputs[i]))
            
    _, predicted = torch.max(q_dist.data, 0)
    return predicted, q_dist


def run_iterative_optimisation(opt, generalist_model, specialist_models, sub_classes, testloader):
    """ index_corrects contains the distribution of the ensemble for every example, and information 
    about whether if the ensemble got the correct prediction (correct = 1, wrong = 0). """
    
    # Make sure models are in evaluation model
    generalist_model.eval() 
    for specialist_i in range(opt.num_specialists): 
        specialist_model =specialist_models[specialist_i]
        specialist_model.eval() 
    
    # RUn iterative algorithm on all examples
    corrects, total, index = 0, 0, 0  
    index_corrects = [] 
    for inputs, labels in testloader: 
        inputs, labels = inputs.to(opt.device), labels.to(opt.device)
        
        num_examples_per_batch = labels.shape[0]
        for example_i in range(num_examples_per_batch): 
            generalist_output = generalist_model(inputs)
            _, baseline_class = torch.max(generalist_output.data, 1)
            specialist_outputs = [] 
            for specialist_i in range(opt.num_specialists): 
                specialist_model = specialist_models[specialist_i]
                specialist_out = specialist_model(inputs)
                specialist_outputs.append(specialist_out)

            ensemble_predicted, q_dist = get_predicted_class(opt.num_classes, baseline_class, generalist_output, specialist_outputs, sub_classes, opt.device)
            if ensemble_predicted == labels[example_i]: 
                corrects += 1 
                index_corrects.append([index, q_dist, 1])
            else: 
                index_corrects.append([index, q_dist, 0])
            total += 1
            index += 1
    return corrects, total, index_corrects 
