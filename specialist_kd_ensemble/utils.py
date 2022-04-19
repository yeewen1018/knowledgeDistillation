import torch
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from itertools import chain

def get_dataset_cifar100():
    """ Loads and normalises the CIFAR-100 dataset from torchvision.datasets. 
    Returns the training and testing dataset. """ 

    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = transformation)
    testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = transformation)
    return trainset, testset


def get_dataloader_cifar100(train_batch_size, test_batch_size):
    """ Loads and normalises the CIFAR-100 dataset from torchvision.datasets. 
    Returns dataloader for the training and testing set. """ 

    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = transformation)
    testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = transformation)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = test_batch_size, shuffle = False)
    return trainloader, testloader


def train_and_evaluate_scratch(trainloader, testloader, model, optimizer, scheduler, criterion, num_epochs, model_path): 
    """ Trains a model from scratch. """
    
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        running_loss, corrects = 0.0, 0
        train_total = 0 
        
        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data 
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
      
          # Zero the parameter gradients 
            optimizer.zero_grad()

          # Forward + Backward + Optimize 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

          # Calculate statistics
            running_loss += loss.item()
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()
            train_total += labels.size(0)
    
        # Evaluation 
        test_correct, test_total = 0, 0 

        with torch.no_grad(): 
            for data in testloader:
                images, labels = data
                if torch.cuda.is_available(): 
                    images, labels = images.cuda(), labels.cuda()

                # Calculate the outputs by running images through the network 
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).cpu().sum().item()
        
        scheduler.step()
        
        if test_correct/test_total > best_test_acc: 
            torch.save(model.state_dict(), model_path)
            best_test_acc = test_correct/test_total

        print(f'[{epoch + 1}] train_loss: {running_loss/train_total}, train_acc: {corrects*100/train_total}, test_acc: {test_correct*100/test_total}')   


def evaluate(model, testloader): 
    test_correct, total = 0, 0 

    with torch.no_grad(): 
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available(): 
                images, labels = images.cuda(), labels.cuda()

        # Calculate the outputs by running images through the network 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predicted == labels).cpu().sum().item()
    
    return test_correct*100/total
    #print(f'Testing accuracy: {test_correct*100/total}')

def get_confusion_matrix(model, testloader): 
    """ Returns the confusion matrix of a model on a set of test data. """
    y_pred, y_true = [], []
    with torch.no_grad(): 
        for data in testloader: 
            images, labels = data 
            if torch.cuda.is_available(): 
                images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)
            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs)
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)
            
    cf_matrix = confusion_matrix(y_true, y_pred)
    return cf_matrix


def plot_corr(cm,size=10):
    """ Plots the correlation matrix from the confusion matrix. """
    
    # Compute the correlation matrix for the received dataframe
    corr = cm.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)


def distillation_loss(student_logits, teacher_logits, hard_labels, temperature, alpha): 
    """ Computes the distillation loss. """
    return nn.KLDivLoss()(F.log_softmax(student_logits/temperature), F.softmax(teacher_logits/temperature)) * (temperature * temperature* alpha)+ F.cross_entropy(student_logits, hard_labels) * (1. - alpha)


def evaluate_specialists_results(dataloader, clusters, specialist_ensemble, generalist_model): 
    """ Evaluate the performance of the specialist ensemble broken down by the number 
    of specialists covering each example in the dataset. """

    # Calculate the number of specialists used for each example in the dataset
    num_specialist_array = [] 
    for i, data in enumerate(dataloader, 0): 
        inputs, labels = data 

        # Calculate the number of specialists used for each example in a batch 
        num_specialist_batch = [] 
        batch_size = labels.shape[0]
        for batch_i in range(batch_size):
            label = labels[batch_i]
            num = 0 
            for index in range(len(clusters)): 
                if label in clusters[index]: 
                    num += 1 
            
            num_specialist_batch.append(num)
        num_specialist_array.append(num_specialist_batch)

    num_covering_specialist = list(set(chain(*num_specialist_array)))
    for i in range(num_covering_specialist): 
        delta_correct, ensemble_corrects, baseline_corrects = 0, 0, 0
        list_index, total = 0, 0 
        for i, data in enumerate(dataloader, 0): 
            inputs, labels = data 
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()
            
            ensemble_output = specialist_ensemble(inputs)
            generalist_output = generalist_model(inputs)

            _, ensemble_predicted = torch.max(ensemble_output.data, 1)
            _, generalist_predicted = torch.max(generalist_output.data, 1)

            batch_size = labels.shape[0]
            for b in range(batch_size): 
                if num_specialist_array[list_index][b] == i: 
                    if ensemble_predicted[b] == labels[b] and generalist_predicted[b] != labels[b]: 
                        delta_correct += 1
                    
                    if ensemble_predicted[b] == labels[b]: 
                        ensemble_corrects += 1
                    
                    if generalist_predicted[b] == labels[b]: 
                        baseline_corrects += 1 

                    total += 1
            list_index += 1
        print(f'Number of specialists covering: {i}, Number of examples: {total}, Delta correct: {delta_correct}, Relative accuracy change: {ensemble_corrects/baseline_corrects}%')







