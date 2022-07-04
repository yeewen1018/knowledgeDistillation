import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn 

def get_cifar100_dataset(cifar100_mean, cifar100_std): 
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std)])
    
    trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = transformation)
    testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = transformation)
    return trainset, testset


def train_and_evaluate_scratch(trainloader, testloader, model, optimizer, scheduler, criterion, num_epochs, model_path, device): 
    lowest_test_loss = 1000.0 
    for epoch in range(num_epochs): 
        running_loss, train_corrects, train_total = 0.0, 0, 0
        model.train() 
        for inputs, labels in trainloader: 
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            train_corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()
            train_total += labels.size(0)
    
        # Evaluation 
        test_corrects, test_total, test_running_loss = evaluate(model, testloader, device)

        scheduler.step()
        if test_running_loss/test_total < lowest_test_loss: 
            torch.save(model.state_dict(), model_path)
            lowest_test_loss = test_running_loss/test_total
 
        print(f'[{epoch + 1}], train_loss: {running_loss/train_total:.4f}, test_loss: {test_running_loss/test_total:.4f}, train_accuracy: {train_corrects*100/train_total:.2f} %, test_accuracy: {test_corrects*100/test_total:.2f} %')


def evaluate(model, testloader, device):   
    criterion = nn.CrossEntropyLoss() 
    test_running_loss, test_corrects, test_total = 0.0, 0, 0 
    model.eval() 
    with torch.no_grad(): 
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device) 

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() 
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            test_corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()
            test_total += labels.size(0)

    return test_corrects, test_total, test_running_loss
