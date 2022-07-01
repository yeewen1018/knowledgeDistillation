import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn 

def get_cifar100_dataset(): 
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
    trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = transformation)
    testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = transformation)
    return trainset, testset


def train_and_evaluate_scratch(trainloader, testloader, model, optimizer, scheduler, criterion, num_epochs, model_path): 
    lowest_test_loss = 1000.0 
    for epoch in range(num_epochs): 
        running_loss, train_corrects, train_total = 0.0, 0, 0
        model.train() 
        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data 
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()
            
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
        test_running_loss, test_corrects, test_total = evaluate(model, testloader)

        scheduler.step()
        if test_running_loss/test_total < lowest_test_loss: 
            torch.save(model.state_dict, model_path)
            lowest_test_loss = test_running_loss/test_total
 
        print('[{}], train_loss: {0:.4f}, test_loss: {0:.4f}, train_accuracy: {0:.2f}, test_accuracy: {0:.2f}'.format(epoch+1, running_loss/train_total, test_running_loss/test_total, 
        train_corrects/train_total, test_corrects/test_total))


def evaluate(model, testloader):   
    criterion = nn.CrossEntropyLoss() 
    test_running_loss, test_corrects, test_total = 0.0, 0, 0 
    model.eval() 
    with torch.no_grad(): 
        for data in testloader:
            inputs, labels = data
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() 
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            test_corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()
            test_total += labels.size(0)

    return test_corrects, test_total, test_running_loss





