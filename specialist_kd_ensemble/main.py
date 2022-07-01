from opts import parse_args
import utils, generalist, specialist, ensemble 

import torch 


if __name__ == '__main__': 
    opt = parse_args() 

    # Load and normalise dataset 
    trainset, testset = utils.get_cifar100_dataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = opt.test_batch_size, shuffle = False)
  
    # Train generalist model / Get pre-trained generalist model  
    generalist_model = generalist.get_generalist_model(opt, trainloader, testloader)

    # Test generalist model 
    test_corrects, test_total, test_running_loss = utils.evaluate(generalist_model, testloader)
    print('Testing accuracy of generalist model: {} %'.format(test_corrects*100/test_total))

    # Get subsets of classes based on covariance matrix of generalist model 
    sub_classes = generalist.get_specialist_subsets(opt, generalist_model, testloader)

    # Train specialist models/ Get pre-trained specialist models 
    specialist_models = specialist.get_specialist_models(opt, trainset, testset, sub_classes)

    # Run iterative algorithm 
    corrects, total, index_corrects = ensemble.run_iterative_optimisation(generalist_model, specialist_models, sub_classes, testloader)
    print("The testing accuracy of the ensemble is {} %".format(corrects*100/total))
