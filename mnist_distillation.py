import torch 
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

class teacherNet(nn.Module): 
    def __init__(self): 
        super(teacherNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)

    def forward(self, x): 
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = 0.5, training = self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.fc3(x)
        return x 

class studentNet(nn.Module): 
    def __init__(self): 
        super(studentNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x): 
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

def distillation_loss(student_logits, teacher_logits, hard_labels, temperature, alpha): 
    return nn.KLDivLoss()(F.log_softmax(student_logits/temperature), F.softmax(teacher_logits/temperature)) * (temperature * temperature* 2.0 * alpha)+ F.cross_entropy(student_logits, hard_labels) * (1. - alpha)


def train_and_evaluate_scratch(trainloader, testloader, model, optimizer): 
    """ To train a model from scratch (without distillation) """
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100): 
        running_loss = 0.0
        corrects = 0 

        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data 
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # Zero the parameter gradients 
            optimizer.zero_grad()

            # Forward pass + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate statistics 
            running_loss += loss.item()
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()
    
    test_correct = 0 
    test_total = 0 
    with torch.no_grad(): 
        for data in testloader: 
            images, labels = data
            if torch.cuda.is_available(): 
                images, labels = images.cuda(), labels.cuda()

            outputs = student_model(images)
            # The class with the highest energy is chosen as the predicted class
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).cpu().sum().item()
    
    print(f'testing accuracy: {test_correct/test_total}')
   

def train_and_evaluate_kd(trainloader, testloader, teacher_model, student_model, optimizer, alpha, temperature): 
    """ To train a student model by distilling knowledge from the teacher model """
    teacher_model.eval()
    student_model.train()

    for epoch in range(100): 
        running_loss = 0.0
        corrects = 0 

        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data 
            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # Zero the parameter gradients 
            optimizer.zero_grad()

            # Forward pass + backward + optimize
            outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs).detach()
            loss = distillation_loss(outputs, teacher_outputs, labels, temperature, alpha)
            loss.backward()
            optimizer.step()

            # Calculate statistics 
            running_loss += loss.item()
            predicted_class = outputs.data.max(1, keepdim = True)[1]
            corrects += predicted_class.eq(labels.data.view_as(predicted_class)).cpu().sum()
    
    test_correct = 0 
    test_total = 0 
    with torch.no_grad(): 
        for data in testloader: 
            images, labels = data
            if torch.cuda.is_available(): 
                images, labels = images.cuda(), labels.cuda()

            outputs = student_model(images)
            # The class with the highest energy is chosen as the predicted class
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).cpu().sum().item()
    
    print(f'alpha: {alpha}, temperature: {temperature}, testing accuracy: {test_correct/test_total}')



if __name__ == '__main__': 

    # Load and normalise the MNIST handwritten digits dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_batch_size = 128
    test_batch_size = 1000

    trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)

    trainloader = torch.utils.data.DataLoader(dataset = trainset, shuffle = True, batch_size = train_batch_size)
    testloader = torch.utils.data.DataLoader(dataset = testset, shuffle = False, batch_size = test_batch_size)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # assuming that we have a pretrained teacher model 
    teacher_model = teacherNet() 
    pretrained_teacher_path = ""
    teacher_model.load_state_dict(torch.load(pretrained_teacher_path))

    student_model = studentNet()
    if torch.cuda.is_available(): 
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()

    optimizer = optim.SGD(student_model.parameters(), lr = 0.1, momentum = 0.9)

    train_and_evaluate_kd(trainloader, testloader, teacher_model, student_model, optimizer, alpha = 0.7, temperature = 20)


