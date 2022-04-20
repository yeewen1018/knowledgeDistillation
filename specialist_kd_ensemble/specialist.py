from venv import create
from sklearn.utils import shuffle
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
import torch.optim as optim 
import random 
import numpy as np 
import utils 


class specialistDataset(torch.utils.data.Dataset):
    """ Customised dataset for a specialist. """
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
    

def create_specialist_dataloader(dataset, cluster, transformation, batch_size, shuffle_boolean):
    """ Returns customised dataloader to train a specialist. 
    The dataset contains 50% examples from the specialist's domain classes and 50% from non-domain classes. """ 

    # Get the index of data that belongs to the specialist's domain 
    indices = []
    for class_label in cluster: 
        index = [i for i in range(len(dataset.targets)) if dataset.targets[i] == class_label]
        indices.append(index)

    # Flatten the list of indices into one list 
    indices = [i for class_indices in indices for i in class_indices]

    # Get another 50% of data from non-domain class
    num_examples = len(indices)
    dustbin_indices = [random.randint(0, len(dataset.targets) - 1) for i in range(0, num_examples * 2) if i not in indices]
    dustbin_indices = dustbin_indices[:num_examples]

    # Combine domain and non-domain (dustbin) indices 
    specialist_indices = indices + dustbin_indices

    # Create dataset 
    specialist_dataset, specialist_dataset_targets = [dataset.data[i] for i in specialist_indices], [dataset.targets[i] for i in specialist_indices]
    s_dataset = specialistDataset(specialist_dataset, specialist_dataset_targets, transform= transformation)
    specialist_dataloader = torch.utils.data.DataLoader(s_dataset, batch_size = batch_size, shuffle = shuffle_boolean)
    return specialist_dataloader



def train_and_evaluate_specialist(trainloader, testloader, teacher_model, specialist_model, optimizer, scheduler, alpha, temperature, num_epochs, cluster, model_path): 
    """ Trains a specialist model. Returns the best testing accuracy achieved by the model. """
    
    best_test_acc = 0.0
    dustbin_class = len(cluster)
    
    teacher_model.eval()
    specialist_model.train()
    
    for epoch in range(num_epochs): 
        running_loss, corrects = 0.0, 0 
        train_total = 0 
        
        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data 
            
            # Correct the labels
            for index in range(labels.shape[0]): 
                label = labels[index]
                if label in cluster: 
                    labels[index] = cluster.index(label)
                else: 
                    labels[index] = dustbin_class
                
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()
                
            # Zero the parameter gradients 
            optimizer.zero_grad()
            
            # Forward 
            outputs = specialist_model(inputs)
            teacher_outputs = teacher_model(inputs).detach()
            
            # Correct the teacher outputs 
            num_batches, num_specialist_classes = outputs.shape[0], len(cluster)
            modified_teacher_outputs = np.zeros((num_batches, num_specialist_classes+1))
            
            for batch_index in range(num_batches): 
                teacher_output = teacher_outputs[batch_index]
                
                for class_index in range(num_specialist_classes): 
                    modified_teacher_outputs[batch_index][class_index] = teacher_output[cluster[class_index]]
                    teacher_output[cluster[class_index]] = 0 
                    
                # Sum the logits for non-domain classes and assign that value as the logit of the dustbin class 
                modified_teacher_outputs[batch_index][dustbin_class] = torch.sum(teacher_output) + np.log(50)
                
            modified_teacher_outputs = torch.tensor(modified_teacher_outputs).float()
            if torch.cuda.is_available(): 
                modified_teacher_outputs = modified_teacher_outputs.cuda()
            
            # Backward + Optimize
            loss = utils.distillation_loss(outputs, modified_teacher_outputs, labels, temperature, alpha) 
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            running_loss += loss.item()
            
          # We need to modify the outputs when calculating the accuracy 
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()
            train_total += labels.size(0)
            
        # Evaluation 
        test_correct, test_total = 0, 0 
        with torch.no_grad(): 
            for data in testloader: 
                images, labels = data 
                
                for j in range(labels.shape[0]): 
                    label = labels[j]
                    if label in cluster: 
                        labels[j] = cluster.index(label)
                    else: 
                        labels[j] = dustbin_class              
                
                if torch.cuda.is_available(): 
                    images, labels = images.cuda(), labels.cuda()

                # Calculate the outputs by running images through the network 
                outputs = specialist_model(images)
    
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).cpu().sum().item()
        
        scheduler.step()
        
        if test_correct/test_total > best_test_acc: 
            torch.save(specialist_model.state_dict(), model_path)
            best_test_acc = test_correct/test_total

        #print(f'[{epoch + 1}] train_loss: {running_loss/train_total}, train_acc: {corrects/train_total}, test_acc: {test_correct/test_total}') 
    return best_test_acc



def train_best_specialist(specialist_model_name, generalist_model, generalist_state_dict, cluster, index, trainset, testset): 
    """ Returns the specialist model trained with the most optimal temperature and alpha values. """
    
    alpha_values = [0.90, 0.95, 0.98]
    temperature_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    specialist_trainloader = create_specialist_dataloader(dataset = trainset, cluster = cluster, batch_size = 128, transformation= transformation, shuffle_boolean= True)
    specialist_testloader = create_specialist_dataloader(dataset= testset, cluster = cluster, transformation= transformation, batch_size= 1024, shuffle_boolean= False)

    best_model_test_acc = 0.0
    for alpha in alpha_values: 
        for temp in temperature_values: 
            specialist_model = torch.hub.load("chenyaofo/pytorch-cifar-models", specialist_model_name, pretrained = False)
            specialist_model.load_state_dict(generalist_state_dict)
            specialist_model.fc = nn.Linear(specialist_model.fc.in_features, len(cluster) + 1) # + 1 for dustbin class
            if torch.cuda.is_available(): 
                specialist_model = specialist_model.cuda()  
            optimizer = optim.SGD(specialist_model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
            num_epochs = 20
            model_path = 'specialists/specialist_kd_model_' + str(index) + '_cifar100.pth'
            best_test_acc = train_and_evaluate_specialist(specialist_trainloader, specialist_testloader, generalist_model, specialist_model, optimizer, scheduler, alpha, temp, num_epochs, cluster, model_path)

            if best_test_acc > best_model_test_acc: 
                best_model_test_acc = best_test_acc
                best_model_state_dict = torch.load(model_path)

    specialist_model.load_state_dict(best_model_state_dict)
    torch.save(best_model_state_dict, model_path)
    return specialist_model 
              

def train_and_get_specialist_models(generalist_model, generalist_model_name, generalist_model_path, clusters, trainset, testset): 
    """ Train specialist models from scratch and returns a list of trained specialist models. """

    specialist_models = [] 
    num_specialists = len(clusters)

    # The specialist models are expected to have the same architecture as the generalist model 
    if torch.cuda.is_available(): 
        generalist_state_dict = torch.load(generalist_model_path)
    else: 
        generalist_state_dict = torch.load(generalist_model_path, map_location= torch.device('cpu'))

    for i in range(num_specialists): 
        specialist_model_name = 'cifar100_' + generalist_model_name 
        model = train_best_specialist(specialist_model_name, generalist_model, generalist_state_dict, clusters[i], i+1, trainset, testset)
        specialist_models.append(model)
    return specialist_models



def get_specialist_models(generalist_model_name, pretrained_specialist_models_dict, clusters):
    """ Returns a list of trained specialist models. """

    specialist_models = [] 
    num_specialists = len(clusters)

    # The specialist models are expected to have the same architecture as the generalist model 
    for i in range(num_specialists): 
        specialist_model_name = "cifar100_" + generalist_model_name
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", specialist_model_name, pretrained = False)
        model.fc = nn.Linear(model.fc.in_features, len(clusters[i]) + 1 )  # +1 for dustbin class 
        if torch.cuda.is_available(): 
            model.load_state_dict(torch.load('specialists/' + pretrained_specialist_models_dict[i]))
            model = model.cuda() 
        else: 
            model.load_state_dict(torch.load('specialists/' + pretrained_specialist_models_dict[i], map_location= torch.device('cpu')))
        specialist_models.append(model)
    return specialist_models 
