import argparse 

def parse_args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train_batch_size', default = 128, type = int, help = 'Batch size for training')
    parser.add_argument('--test_batch_size', default = 1, type = int, help = 'Batch size for inference')
    parser.add_argument('--cifar100_mean', default = (0.5071, 0.4867, 0.4408), type= tuple, help = "Mean for CIFAR-100 dataset")
    parser.add_argument('--cifar100_std', default = (0.2675, 0.2565, 0.2761), type = tuple, help = "Standard deviation for CIFAR-100 dataset")
    parser.add_argument('--model_type', default = 'resnet20', type = str, help = 'Model architecture')
    parser.add_argument('--pretrained_generalist_path', default='models/teacher_model_cifar100.pth', type = str, help = 'Path to pretrained generalist model')
    parser.add_argument('--lr', default = 0.001, type =float, help = 'Learning rate')
    parser.add_argument('--momentum', default = 0.9, type = float, help = 'Momentum')
    parser.add_argument('--nesterov', default=True, type = bool, help='Use Nesterov or not')
    parser.add_argument('--weight_decay', default=0.0005, type = float, help = 'Weight decay')
    parser.add_argument('--lr_step_size', default = 20, type = int, help = 'Step size for learning rate scheduler')
    parser.add_argument('--lr_scheduler_gamma', default = 0.1, type = float, help = 'Gamma for learning rate scheduler') 
    parser.add_argument('--generalist_num_train_epochs', default = 10, type = int, help = 'Number of epochs to train the generalist model')
    parser.add_argument('--num_classes', default = 100, type = int, help = "Number of classes")
    parser.add_argument('--predefined_specialist_subsets', default= True, type = bool, help = 'Use predefined specialist subsets')
    parser.add_argument('--num_specialists', default = 11, type = int, help = 'Number of specialists')

    args = parser.parse_args() 
    return args 
