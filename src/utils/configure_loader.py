import sys
sys.path.append('..')


def load_configure(model_name, dataset_name):
    model_dataset_name = f'{model_name}_{dataset_name}'
    if model_dataset_name == 'simcnn_cifar10':
        from configures.simcnn_cifar10 import Configures
    elif model_dataset_name == 'simcnn_svhn':
        from configures.simcnn_svhn import Configures
    elif model_dataset_name == 'rescnn_cifar10':
        from configures.rescnn_cifar10 import Configures
    elif model_dataset_name == 'rescnn_svhn':
        from configures.rescnn_svhn import Configures
    elif model_dataset_name == 'incecnn_cifar10':
        from configures.incecnn_cifar10 import Configures
    elif model_dataset_name == 'incecnn_svhn':
        from configures.incecnn_svhn import Configures
    else:
        raise ValueError()
    configs = Configures()
    return configs
