import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 

import utils

def get_predicted_class(inputs, generalist_class, generalist_output, q_dist, specialist_models, clusters, batch_number): 
    """ Iteratively finds the predicted class where the generalist model and all specialist models agree on. """
    
    num_specialists = len(clusters)
    optimizer = optim.Adam([q_dist], lr = 0.005)
    criterion = nn.KLDivLoss()
    loss = criterion(F.log_softmax(q_dist), F.softmax(generalist_output))
    previous_loss = 10000.0
    
    # It is good to have a threshold here (i.e, 1e-3), because for some examples, 
    # the loss value keeps decreasing and the while loop becomes endless. 
    # There are definitely better ways to do this optimisation, but both these conditions work sufficiently fine.  
    while previous_loss> loss and loss.item()> 1e-3:      
        
        for i in range(num_specialists): 
            if generalist_class in clusters[i]: 
                cluster = clusters[i]
                model = specialist_models[i]
                specialist_output = model(inputs)
            
                modified_q_dist = np.zeros(len(cluster) + 1)
                q_dist_copy = q_dist.clone()
                for index in range(len(cluster)):
                    modified_q_dist[index] = q_dist_copy[cluster[index]]
                    q_dist_copy[cluster[index]] = 0 
                
                modified_q_dist[len(cluster)] = torch.sum(q_dist_copy)
                modified_q_dist = torch.tensor(modified_q_dist)
                if torch.cuda.is_available(): 
                    modified_q_dist = modified_q_dist.cuda()
            
                loss += criterion(F.log_softmax(modified_q_dist), F.softmax(specialist_output[batch_number]))
        
        previous_loss = loss
        loss.backward(retain_graph = True)
        optimizer.step()
        optimizer.zero_grad()
        loss = criterion(F.log_softmax(q_dist), F.softmax(generalist_output))
        
        # Get predicted class 
    _, predicted = torch.max(q_dist.data, 0)
    return predicted     


def get_optimal_q_iteratively(generalist_model, specialist_models, clusters, dataloader): 
    """ Finds the optimal q distribution for an example. """
    corrects, total = 0, 0 
    for i, data in enumerate(dataloader, 0): 
        inputs, labels = data 
        if torch.cuda.is_available(): 
            inputs, labels = inputs.cuda(), labels.cuda()

        # Get predicted class from the generalist model 
        generalist_output = generalist_model(inputs)
        _, generalist_predicted = torch.max(generalist_output.data, 1)

        # For each sample in the batch, iteratively find the optimal probability distribution 
        num_batches, num_classes = generalist_output.shape[0], generalist_output.shape[1]
        for batch in range(num_batches): 
            q_dist = torch.randn(num_classes, requires_grad = True, device = "cuda")
            predicted_class = get_predicted_class(inputs, generalist_predicted[batch], generalist_output[batch], q_dist, specialist_models, clusters, batch)

            if predicted_class == labels[batch]: 
                corrects += 1 
            total += 1
        
        print(f'corrects: {corrects}, totals: {total}')


# Train a linear layer 
class MyEnsemble(nn.Module): 
    """ Construct an ensemble that takes in the summed output of the generalist model 
    and all specialist models as input to train its linear layer. """
    def __init__(self, generalist_model, specialist_models, clusters): 
        super(MyEnsemble, self).__init__()
        self.generalist_model = generalist_model 
        self.specialist_models = specialist_models 
        self.clusters = clusters 
        self.num_specialists = len(self.clusters)

        # Freeze the parameters of the generalist and specialist models 
        for param in self.generalist_model.parameters(): 
            param.requires_grad = False 

        for i in range(self.num_specialists): 
            model = self.specialist_models[i]
            for param in model.parameters(): 
                param.requires_grad = False 

        self.linear_classifier = nn.Linear(100, 100)


    def forward(self, x):       
        # Get output from generalist model 
        output = self.generalist_model(x.clone())
        output = output.view(output.size(0), -1)
        
        num_batches, num_classes = output.shape[0], output.shape[1]
        
        # Go through all specialist models 
        for i in range(self.num_specialists): 
            model = self.specialist_models[i]
            out = model(x.clone())
            out = out.view(out.size(0), -1)
            dustbin_class = len(self.clusters[i])
            out_modified = self.get_specialist_logits(out, num_batches, num_classes, dustbin_class, self.clusters[i])
            output += out_modified 
        
        # Go through the linear layer 
        output = F.relu(self.linear_classifier(output))
        return output 

    def get_specialist_logits(self, logits, num_batches, num_classes, dustbin_class, cluster): 
        logits_modified = np.zeros((num_batches, num_classes))
        logits = logits.detach().cpu().numpy()
        
        for mini_batch in range(num_batches): 
            logits_modified[mini_batch] = (logits[mini_batch][dustbin_class])/ (100 - dustbin_class)
        
            for c in range(len(cluster)): 
                logits_modified[mini_batch][cluster[c]] = logits[mini_batch][c]

        logits_modified = torch.tensor(logits_modified)
        if torch.cuda.is_available(): 
            logits_modified = logits_modified.cuda()
        return logits_modified 


def get_ensemble_linear_layer(generalist_model, specialist_models, clusters, pretrained_ensemble_path, trainloader, testloader): 
    """ Returns a trained specialist ensemble model with linear layer. """
    specialist_ensemble = MyEnsemble(generalist_model, specialist_models, clusters)
    if pretrained_ensemble_path is not None: 
        if torch.cuda.is_available(): 
            state_dict = torch.load(pretrained_ensemble_path)
            specialist_ensemble = specialist_ensemble.cuda() 
        else: 
            state_dict = torch.load(pretrained_ensemble_path, map_location= torch.device('cpu'))
        #specialist_ensemble.load_state_dict(state_dict)
        model_dict = specialist_ensemble.state_dict()
        state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        specialist_ensemble.load_state_dict(model_dict)

    else: 
        if torch.cuda.is_available(): 
            specialist_ensemble = specialist_ensemble.cuda()
        optimizer = optim.SGD(specialist_ensemble.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 80, gamma = 0.1)
        criterion = nn.CrossEntropyLoss()
        num_epochs = 100 

        utils.train_and_evaluate_scratch(trainloader, testloader, specialist_ensemble, optimizer, scheduler, criterion, num_epochs, 'specialists/specialist_ensemble_cifar100.pth')
        if torch.cuda.is_available(): 
            specialist_ensemble.load_state_dict(torch.load('specialists/specialist_ensemble_cifar100.pth'))
        else: 
            specialist_ensemble.load_state_dict(torch.load('specialists/specialist_ensemble_cifar100.pth', map_location= torch.device('cpu')))
    return specialist_ensemble


    