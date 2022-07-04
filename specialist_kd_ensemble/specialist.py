import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import random 
import numpy as np 

import utils 

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
    torch_specialist_dataset = specialistDataset(specialist_dataset, specialist_dataset_targets, subset_class, transform = transformation)
    if not dataloader: 
        return torch_specialist_dataset
    specialist_dataloader = torch.utils.data.DataLoader(torch_specialist_dataset, batch_size = batch_size, shuffle = shuffle)
    return specialist_dataloader


class specialistDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, sub_class, transform=None):
        super(specialistDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        self.sub_class = sub_class
        self.dustbin_class = len(self.sub_class)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.images[idx]
        if self.transform: 
            sample = self.transform(sample)
        label = self.labels[idx]
        if label in self.sub_class: 
            label = self.sub_class.index(label)
        else: 
            label = self.dustbin_class
        return sample, label 


def get_specialist_models(opt, trainset, testset, sub_classes): 
    specialist_models = [] 
    model_name = "cifar100_" + opt.model_type
    
    generalist_state_dict = torch.load(opt.pretrained_generalist_path, map_location = opt.device)
    for i in range(opt.num_specialists): 
        sub_class = sub_classes[i]
        specialist_model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained = False)
        specialist_model.load_state_dict(generalist_state_dict)
        specialist_model.fc = nn.Linear(specialist_model.fc.in_features, len(sub_class)+1)
        specialist_model.to(opt.device)
        specialist_traindataloader = create_specialist_dataset(trainset, sub_class, batch_size = 32, train = True, dataloader = True, shuffle = True)
        specialist_testdataloader = create_specialist_dataset(testset, sub_class, opt.test_batch_size, train = False, dataloader = True, shuffle = False)
        model_path = 'models/specialist_' + str(i+1) + '.pth'

        if not opt.predefined_specialist_subsets: 
            print("Training specialist {}".format(i+1))
            print("=========================================================================================================")
            criterion = nn.CrossEntropyLoss() 
            optimizer = optim.SGD(specialist_model.parameters(), lr = opt.lr, nesterov = opt.nesterov, momentum = opt.momentum, weight_decay= opt.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = opt.lr_step_size, gamma = opt.lr_scheduler.gamma)
            num_epochs = 20 
            utils.train_and_evaluate_scratch(specialist_traindataloader, specialist_testdataloader, specialist_model, optimizer, scheduler, criterion, num_epochs, model_path, opt.device)
        
        specialist_model.load_state_dict(torch.load(model_path, map_location = opt.device))
        specialist_models.append(specialist_model)
        test_corrects, test_total, _ = utils.evaluate(specialist_model, specialist_testdataloader, opt.device)
        print(f'Testing accuracy of specialist model {i+1} is {test_corrects*100/test_total:.2f}%')
        print("==============================================================================================================")
    return specialist_models


def print_specialist_subsets(sub_classes, label_names): 
    for i in range(len(sub_classes)): 
        sub_class = sub_classes[i]
        sub_class_names = [] 
        for sub_class_i in range(len(sub_class)): 
            sub_class_names.append(label_names[sub_class[sub_class_i]])
        print(f'Subset of specialist {i+1}: {sub_class_names}')
