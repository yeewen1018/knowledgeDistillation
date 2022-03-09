import torch 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 

# Define a convolutional neural network for the student model 
class studentNet(nn.Module): 
    def __init__(self, num_channels): 
        super(studentNet, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride = 1, padding =1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # Define fully connected layers to transform the output feature maps of the convolutional layers
        self.fc1 = nn.Linear(4*4*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 10)
    
    def forward(self, x): 
        x = self.bn1(self.conv1(x))     
        x = F.relu(F.max_pool2d(x, 2))  
        x = self.bn2(self.conv2(x))     
        x = F.relu(F.max_pool2d(x, 2))  
        x = self.bn3(self.conv3(x))    
        x = F.relu(F.max_pool2d(x, 2))  

        # Flatten the image 
        x = x.view(-1, 4*4*self.num_channels*4)

        x = F.dropout(F.relu(self.fcbn1(self.fc1(x))), p = 0.7, training = self.training)
        x = self.fc2(x)
    
        return x 


def distillation_loss(student_logits, teacher_logits, hard_labels, temperature, alpha): 
    return nn.KLDivLoss()(F.log_softmax(student_logits/temperature), F.softmax(teacher_logits/temperature)) * (temperature * temperature* 2.0 * alpha)+ F.cross_entropy(student_logits, hard_labels) * (1. - alpha)


def train_and_evaluate_from_scratch(trainloader, testloader,student_model, optimizer): 
    student_model.train()
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 80, gamma = 0.1)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(200): 
        running_loss = 0.0
        corrects = 0 
        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data 
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients 
            optimizer.zero_grad() 

            # forward + backward + optimize 
            outputs = student_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

          # Calculate statistics 
            running_loss += loss.item()
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()


      # Test out the student model 
    correct = 0
    total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available(): 
                images, labels = images.cuda(), labels.cuda()
        
            # calculate outputs by running images through the network
            outputs = student_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum().item()

    print(f'Testing accuracy: {correct/ total} ')


def train_and_evaluate_kd(trainloader, testloader, teacher_model, student_model, optimizer, alpha, temperature): 
    teacher_model.eval()
    student_model.train()
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 80, gamma = 0.1)
    for epoch in range(200): 
        running_loss = 0.0
        corrects = 0 
        scheduler.step()
        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data 
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients 
            optimizer.zero_grad() 

            # forward + backward + optimize 
            outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs).detach()
            loss = distillation_loss(outputs, teacher_outputs, labels, temperature = temperature, alpha = alpha)
            loss.backward()
            optimizer.step()

          # Calculate statistics 
            running_loss += loss.item()
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()

      # Test out the student model 
    correct = 0
    total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available(): 
                images, labels = images.cuda(), labels.cuda()
        
            # calculate outputs by running images through the network
            outputs = student_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum().item()

    print(f'alpha: {alpha}, temperature: {temperature}, testing accuracy: {correct/total} ')

import resnet, densenet

if __name__ == '__main__': 
    # Load and normalise CIFAR-10 dataset 
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_batch_size = 128
    test_batch_size = 1000

    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = test_batch_size, shuffle = False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    teacher_model_name = "" # ['resnet18', 'resnet34', 'resnet50', 'densenet121', 'densenet169']
    if teacher_model_name == 'resnet18': 
        teacher_model = resnet.resnet18()
    elif teacher_model_name == 'resnet34': 
        teacher_model = resnet.resnet34()
    elif teacher_model_name == 'resnet50': 
        teacher_model = resnet.resnet50()
    elif teacher_model_name == 'densenet121': 
        teacher_model = densenet.densenet121()
    elif teacher_model_name == 'densenet161': 
        teacher_model = densenet.densenet161()
    elif teacher_model_name == 'densenet169': 
        teacher_model = densenet.densenet169()

    teacher_model_pretrained_path = ""
    teacher_model.load_state_dict(torch.load(teacher_model_pretrained_path))

    student_model_name = "" # ['studentNet', 'resnet18']
    if student_model_name == 'studentNet': 
        student_model = studentNet(num_channels= 32)
    elif student_model_name == 'resnet18': 
        student_model = resnet.resnet18()

    if torch.cuda.is_available(): 
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()

    optimizer = optim.Adam(student_model.parameters(), lr = 0.01 )

    train_and_evaluate_kd(trainloader, testloader, teacher_model, student_model, optimizer, alpha = 0.9, temperature = 6)
    

    

