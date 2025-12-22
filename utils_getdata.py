from torchvision.transforms import ToTensor, transforms
from torchvision import datasets
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.sample_dirichlet import clients_indices
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, get_class_num
import copy

def my_get_data(args):
    # Load data
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset_name == 'Cifar10':
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    elif args.dataset_name == 'Cifar100':
        data_local_training = datasets.CIFAR10(args.path_cifar100, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR10(args.path_cifar100, train=False, transform=transform_all)
    else:
        exit("no suit dataset as {}".format(args.dataset_name))

    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    indices2data = Indices2Dataset(data_local_training)
    return list_client2indices, original_dict_per_client, data_global_test, indices2data