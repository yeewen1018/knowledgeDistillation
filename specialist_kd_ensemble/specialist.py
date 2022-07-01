import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import random 
import numpy as np 

 
def create_specialist_dataset(dataset, subset_class, batch_size, train, dataloader, shuffle = True): 
    """ Returns dataset enriched with examples that the specialist model specializes in. """
    
    # Gather all images that belong to specialist's sub-class 
    num_examples = len(dataset.targets)
    
    subset_indices = [] 
    for label in subset_class: 
        indices = [i for i in range(num_examples) if dataset.targets[i] == label]
        subset_indices.append(indices)
        
    # Flatten the list of lists into one list 
    subset_indices = [item for sublist in subset_indices for item in sublist]
    
    # Get training data from dustbin class
    if train: 
        num_dustbin_examples = 500
    else: 
        num_dustbin_examples = 100 
        
    random_indices = np.random.randint(0, num_examples - 1, num_dustbin_examples * 5)
    dustbin_indices = [] 
    for index in random_indices: 
        if index in subset_indices: 
            continue
        else: 
            dustbin_indices.append(index)          
    random.shuffle(dustbin_indices)
    dustbin_indices = dustbin_indices[:num_dustbin_examples]
    
    # Combine examples from specialised subset and dustbin class 
    indices = subset_indices + dustbin_indices 
    
    # Create dataset 
    specialist_dataset, specialist_dataset_targets = [dataset.data[i] for i in indices], [dataset.targets[i] for i in indices]
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    torch_specialist_dataset = specialistDataset(specialist_dataset, specialist_dataset_targets, transform = transformation)
    if not dataloader: 
        return torch_specialist_dataset
    
    specialist_dataloader = torch.utils.data.DataLoader(torch_specialist_dataset, batch_size = batch_size, shuffle = shuffle)
    return specialist_dataloader



class specialistDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        super(specialistDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.images[idx]
        if self.transform: 
            sample = self.transform(sample)
        label = self.labels[idx]
        return sample, label 


def train_and_evaluate_specialist(trainloader, testloader, specialist_model, optimizer, scheduler, criterion, num_epochs, sub_class, model_path): 
    
    lowest_test_loss = 10.0
    dustbin_class = len(sub_class)
    
    for epoch in range(num_epochs): 
        running_loss, corrects, train_total = 0.0, 0, 0 
        specialist_model.train()

        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data 
            
            # Correct the labels
            for index in range(labels.shape[0]): 
                label = labels[index]
                if label in sub_class: 
                    labels[index] = sub_class.index(label)
                else: 
                    labels[index] = dustbin_class
                
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()
                
            # Zero the parameter gradients 
            optimizer.zero_grad()
            outputs = specialist_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            running_loss += loss.item()
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()
            train_total += labels.size(0)
            
        # Evaluation 
        test_correct, test_total, test_running_loss = evaluate_specialist(specialist_model, testloader, sub_class)
        
        scheduler.step()
        if test_running_loss/test_total < lowest_test_loss: 
            torch.save(specialist_model.state_dict(), model_path)
            lowest_test_loss = test_running_loss/test_total
        
        #print('[{}], train_loss: {0:.4f}, test_loss: {0:.4f}, train_accuracy: {0:.2f}, test_accuracy: {0:.2f}'.format(epoch+1, running_loss/train_total, test_running_loss/test_total, 
        #            corrects/train_total, test_correct/test_total))


def evaluate_specialist(specialist_model, testloader, sub_class): 
    test_correct, test_total, test_running_loss = 0, 0, 0.0 
    dustbin_class = len(sub_class)
    criterion = nn.CrossEntropyLoss() 
    specialist_model.eval() 
    with torch.no_grad(): 
        for data in testloader: 
            images, labels = data            
            for j in range(labels.shape[0]): 
                label = labels[j]
                if label in sub_class: 
                    labels[j] = sub_class.index(label)
                else: 
                    labels[j] = dustbin_class              
                
            if torch.cuda.is_available(): 
                images, labels = images.cuda(), labels.cuda()

            outputs = specialist_model(images)
            test_loss = criterion(outputs, labels)
            test_running_loss += test_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).cpu().sum().item() 
    
    return test_correct, test_total, test_running_loss 


def get_specialist_models(opt, trainset, testset, sub_classes): 
    specialist_models = [] 
    model_name = "cifar100_" + opt.model_type
    if not opt.predefined_specialist_subsets:     
        print("Training specialist models. ")
        print("==============================================================================")
        generalist_state_dict = torch.load(opt.pretrained_generalist_path, map_location = torch.device('cpu'))
        for i in range(opt.num_specialists): 
            sub_class = sub_classes[i]
            specialist_model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained = False)
            specialist_model.load_state_dict(generalist_state_dict)
            specialist_model.fc = nn.Linear(specialist_model.fc.in_features, len(sub_class) + 1)
            if torch.cuda.is_available(): 
                specialist_model = specialist_model.cuda() 
            
            criterion = nn.CrossEntropyLoss() 
            optimizer = optim.SGD(specialist_model.parameters(), lr = opt.lr, nesterov = opt.nesterov, momentum = opt.momentum, weight_decay = opt.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = opt.lr_step_size, gamma = opt.lr_scheduler_gamma)
            num_epochs = 20 
            model_path = 'models/specialist_' + str(i+1) + '.pth'

            specialist_train_batch_size = 32
            specialist_traindataloader = create_specialist_dataset(trainset, sub_class, specialist_train_batch_size, train = True, dataloader = True, shuffle = True)
            specialist_testdataloader = create_specialist_dataset(testset, sub_class, opt.test_batch_size, train = False, dataloader = True, shuffle = False)

            train_and_evaluate_specialist(specialist_traindataloader, specialist_testdataloader, specialist_model, optimizer, scheduler, criterion, num_epochs, sub_class, model_path)
            print("Testing accuracy of specialist model {} is {} %".format(i+1, evaluate_specialist(specialist_model, specialist_testdataloader, sub_class)))
            print("==============================================================================================================")
            specialist_models.append(specialist_model)
    
    else: 
        print("Using pre-trained specialist models")
        for i in range(opt.num_specialists): 
            sub_class = sub_classes[i]
            specialist_model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained = False)
            specialist_model.fc = nn.Linear(specialist_model.fc.in_features, len(sub_class) + 1)
            model_path = 'models/specialist_' + str(i+1) + '.pth'
            specialist_model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
            if torch.cuda.is_available(): 
                specialist_model = specialist_model.cuda() 
            specialist_models.append(specialist_model)
    
    return specialist_models

