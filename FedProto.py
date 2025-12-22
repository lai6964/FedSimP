from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from options import args_parser
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, get_class_num
from Dataset.sample_dirichlet import clients_indices
from Dataset.Gradient_matching_loss import match_loss
import numpy as np
from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from Model.Resnet8 import ResNet_cifar
from tqdm import tqdm
import copy
import torch
import random
import torch.nn as nn
import time
from Dataset.param_aug import DiffAugment
from utils import compute_global_protos, _get_prototypes_by_labels, compute_mean_and_variance, cluster_protos_by_Truepredict, compute_global_protos
from collections import defaultdict


class Global(object):
    def __init__(self,
                 num_classes: int,
                 device: str,
                 args,
                 num_of_feature):
        self.device = device
        self.num_classes = num_classes
        self.num_of_feature = num_of_feature
        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # fedavg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.syn_model.state_dict()


class Local(object):
    def __init__(self,
                 data_client,
                 class_list: int):
        args = args_parser()

        self.data_client = data_client

        self.device = args.device
        self.class_compose = class_list

        self.criterion = CrossEntropyLoss().to(args.device)
        self.loss_mse = nn.MSELoss().to(args.device)
        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False,
                                        num_classes=args.num_classes).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])



    def get_protos(self):
        protos = defaultdict(list)
        trainloader = DataLoader(dataset=self.data_client,
                                 batch_size=args.batch_size_local_training,
                                 shuffle=True)
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                # x = self.transform_train(x)
                rep, output = self.local_model(x)

                pred_probs = torch.softmax(output, dim=1)  # 所有类别的预测概率
                pred_classes = torch.argmax(pred_probs, dim=1)  # 预测的类别
                true_class_probs = pred_probs[torch.arange(len(y)), y]  # 真实类别的概率
                pred_class_probs = torch.max(pred_probs, dim=1)[0]  # 预测类别的概率

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    pred_c = pred_classes[i].item()
                    protos[y_c].append({
                        'true_class': y_c,  # 真实类别
                        'pred_class': pred_c,  # 预测类别
                        'true_class_prob': true_class_probs[i].item(),  # 真实类别的概率
                        'pred_class_prob': pred_class_probs[i].item(),  # 预测类别的概率
                        'is_correct': (y_c == pred_c),  # 预测是否正确
                        # 'is_hard': (true_class_probs[i].item() < self.hard_thread),  # 是否困难样本
                        'confidence': pred_class_probs[i].item(),  # 置信度（预测类别的概率）
                        'feature': rep[i, :].detach().data,
                        'all_probs': pred_probs[i, :].detach().data  # 所有类别的概率
                    })
        return protos

    def local_train(self, args, global_params, global_protos):

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images = self.transform_train(images)
                features, outputs = self.local_model(images)
                loss = self.criterion(outputs, labels)


                if global_protos is not None:
                    proto_teacher = _get_prototypes_by_labels(features, labels, global_protos)
                    loss += self.loss_mse(features, proto_teacher) * args.lamda_proto

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.local_model.state_dict()


def FedProto(args):
    # args = args_parser()
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'
        'lr_net:{lr_net}, lr_feature:{lr_feature}, num_of_feature:{num_of_feature}\n '
        'match_epoch:{match_epoch}, re_training_epoch:{crt_epoch}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            lr_net=args.lr_net,
            lr_feature=args.lr_feature,
            num_of_feature=args.num_of_feature,
            match_epoch=args.match_epoch,
            crt_epoch=args.crt_epoch))
    random_state = np.random.RandomState(args.seed)
    # Load data
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args,
                          num_of_feature=args.num_of_feature)
    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)
    re_trained_acc = []
    global_protos = None
    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []
        list_proto_features_local_data, list_proto_vars_local_data, list_proto_nums_local_data = [], [], []
        # local training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])
            # local update
            local_params = local_model.local_train(args, copy.deepcopy(global_params),global_protos)
            list_dicts_local_params.append(copy.deepcopy(local_params))

            local_protos = local_model.get_protos()
            protos, vars, samples_T_num = cluster_protos_by_Truepredict(local_protos, False)
            list_proto_features_local_data.append(copy.deepcopy(protos))
            list_proto_vars_local_data.append(copy.deepcopy(vars))
            list_proto_nums_local_data.append(samples_T_num)
        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        global_protos, global_vars = compute_global_protos(list_proto_features_local_data,list_proto_vars_local_data,list_proto_nums_local_data)
        # global eval
        one_re_train_acc = global_model.global_eval(fedavg_params, data_global_test, args.batch_size_test)
        re_trained_acc.append(one_re_train_acc)
        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
        if r % 10 == 0:
            print(re_trained_acc)
    with open("{}_{}_FedProto.txt".format(args.dataset_name, int(1.0/args.imb_factor)),"w") as f:
        for i, acc in enumerate(re_trained_acc):
            f.write("epoch_"+str(i)+":"+str(acc)+"/n")
    print(re_trained_acc)


if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    args.lamda_proto = 1.0
    args.dsa = False
    FedProto(args)


