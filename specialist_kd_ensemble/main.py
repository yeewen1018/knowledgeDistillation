from opts import parse_args
import utils, generalist, specialist, ensemble 

import torch 


if __name__ == '__main__': 
    opt = parse_args() 
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load and normalise dataset 
    trainset, testset = utils.get_cifar100_dataset(opt.cifar100_mean, opt.cifar100_std)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = opt.test_batch_size, shuffle = False)
  
    # Train generalist model / Get pre-trained generalist model  
    generalist_model = generalist.get_generalist_model(opt, trainloader, testloader)

    # Test generalist model 
    test_corrects, test_total, test_running_loss = utils.evaluate(generalist_model, testloader, opt.device)
    print('Testing accuracy of generalist model: {0:.2f} %'.format(test_corrects*100/test_total))

    # Get subsets of classes based on covariance matrix of generalist model 
    sub_classes = generalist.get_specialist_subsets(opt, generalist_model, testloader)
    specialist.print_specialist_subsets(sub_classes, testset.classes)

    # Train specialist models/ Get pre-trained specialist models 
    specialist_models = specialist.get_specialist_models(opt, trainset, testset, sub_classes)

    # Run iterative algorithm 
    corrects, total, index_corrects = ensemble.run_iterative_optimisation(opt, generalist_model, specialist_models, sub_classes, testloader)
    print("The testing accuracy of the ensemble is {0:.2f} %".format(corrects*100/total))
