import utils, generalist, specialist, ensemble 


if __name__ == '__main__': 

    # Load and normalise CIFAR-100 dataset.  
    trainloader, testloader = utils.get_dataloader_cifar100(train_batch_size = 128, test_batch_size = 1024)

    # Train/Get the generalist model 
    generalist_model = generalist.get_generalist_model_cifar100(model_name = 'resnet20', pretrained_model_path = 'specialists/generalist_model_cifar100.pth', trainloader = trainloader, testloader = testloader)

    # Get confusable classes from the generalist model
    clusters = generalist.get_confusable_classes(model = generalist_model, testloader = testloader)

    # Train/Get specialist models 
    num_specialists = len(clusters)
    pretrained_specialists = True

    specialist_models_dict = [] 
    for i in range(num_specialists): 
        specialist_model_path = 'specialist_kd_model_' + str(i+1) + '_cifar100.pth'
        specialist_models_dict.append(specialist_model_path)

    if pretrained_specialists:
        specialist_models = specialist.get_specialist_models('resnet20', specialist_models_dict, clusters) 
    else: 
        trainset, testset = utils.get_dataset_cifar100()
        specialist_models = specialist.train_and_get_specialist_models(generalist_model, 'resnet20', 'specialists/generalist_model_cifar100.pth', clusters, trainset, testset)

    # Ensemble 
    iterative = False
    pretrained_linear = True 
    
    if iterative:
        # Iteratively find the optimal probability distribution q 
        specialist_ensemble = ensemble.get_optimal_q_iteratively(generalist_model, specialist_models, clusters, testloader)
    else: 
        # Train a linear layer that takes in the output of the generalist model and all specialist models 
        specialist_ensemble = ensemble.get_ensemble_linear_layer(generalist_model, specialist_models, clusters, 'specialists/specialist_ensemble_cifar100.pth', None, None) 

    # Evaluate performance of models 
    generalist_test_acc = utils.evaluate(generalist_model, testloader)
    print(f'Accuracy of baseline (generalist) model: {generalist_test_acc}')
    if not iterative: 
        specialist_test_acc = utils.evaluate(specialist_ensemble, testloader)
        print(f'Accuracy of specialist ensemble: {specialist_test_acc}')

        # Relative accuracy change broken down by number of specialists 
        utils.evaluate_specialists_results(testloader, clusters, specialist_ensemble, generalist_model)