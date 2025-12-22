import torch
from collections import defaultdict

def _get_prototypes_by_labels(rep, labels, protos_dict):
    """根据标签获取对应的原型"""
    proto_result = rep.clone().detach()
    for i, label in enumerate(labels):
        label_val = label.item()
        if label_val in protos_dict:
            proto_result[i, :] = protos_dict[label_val].data
    return proto_result


def compute_mean_and_variance(protos, probs):
    weighted_prototypes = {} # 存储均值
    weighted_variances = {}  # 存储协方差矩阵
    sample_counts = {}  # 存储每个类别的样本数量

    for y_c in protos.keys():
        feature_list = protos[y_c]
        prob_list = probs[y_c]
        sample_counts[y_c] = len(feature_list)  # 记录样本数量
        # 将特征列表和概率列表转换为张量,计算加权平均原型: sum(proto * prob) / sum(prob)
        features_tensor = torch.stack(feature_list)  # [n_samples, feature_dim]
        probs_tensor = torch.tensor(prob_list, device=features_tensor.device)  # [n_samples]
        total_prob = torch.sum(probs_tensor)  # 标量
        # 计算加权特征和
        probs_expanded = probs_tensor.unsqueeze(1).expand_as(features_tensor)  # [n_samples, feature_dim]
        weighted_features_sum = torch.sum(features_tensor * probs_expanded, dim=0)  # [feature_dim]
        weighted_proto = weighted_features_sum / (total_prob + 1e-8)  # [feature_dim] 计算加权平均原型
        weighted_prototypes[y_c] = weighted_proto

        # 计算加权方差
        squared_deviations = (features_tensor - weighted_proto.unsqueeze(0)) ** 2  # [n_samples, feature_dim]
        weighted_squared_deviations = squared_deviations * probs_expanded  # 对平方偏差进行加权
        variance = torch.sum(weighted_squared_deviations, dim=0) / (total_prob + 1e-8) + 1e-6 # [feature_dim]
        weighted_variances[y_c] = variance
    return weighted_prototypes, weighted_variances, sample_counts

def cluster_protos_by_Truepredict(protos_list, using_true_samples_only=True):
    """按照真实类别中的正确预测聚类"""
    protos = defaultdict(list)
    probs = defaultdict(list)
    for y_c, dict_list in protos_list.items():
        for items in dict_list:
            if using_true_samples_only and not items['is_correct']:
                continue
            protos[y_c].append(items['feature'])
            probs[y_c].append(items['true_class_prob'])

    prototypes, variances, counts = compute_mean_and_variance(protos, probs)
    return prototypes, variances, counts

def compute_global_protos(uploaded_protos, uploaded_vars, uploaded_nums):
    """
    计算全局均值和方差
    使用加权平均，权重为每个客户端在每个类别上的样本数量
    """
    # 初始化全局统计信息
    global_protos = {}
    global_vars = {}

    # 获取所有类别
    all_classes = set()
    for protos in uploaded_protos:
        all_classes.update(protos.keys())

    # 对每个类别分别计算全局均值和方差
    for class_id in all_classes:
        # 收集所有客户端中该类别的信息
        class_means = []
        class_variances = []
        class_weights = []  # 权重（样本数量）

        for i, client_protos in enumerate(uploaded_protos):
            if class_id in client_protos:
                class_means.append(client_protos[class_id])
                class_variances.append(uploaded_vars[i][class_id])
                class_weights.append(uploaded_nums[i][class_id])

        if len(class_means)==0:  # 如果没有客户端有这个类别
            print("error when cluster global protos for class_id {}".format(class_id))
            continue

        # 转换为张量
        means_tensor = torch.stack(class_means)
        variances_tensor = torch.stack(class_variances)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=means_tensor.device)

        # 计算加权均值
        total_weight = torch.sum(weights_tensor)
        weighted_means = means_tensor * weights_tensor.unsqueeze(1)
        global_mean = torch.sum(weighted_means, dim=0) / total_weight

        # 计算加权方差
        # 使用合并方差公式: σ²_total = Σ[n_i × (σ_i² + (μ_i - μ_total)²)] / Σn_i
        # 计算每个客户端均值与全局均值的偏差平方
        deviations = means_tensor - global_mean.unsqueeze(0)  # [m, d]
        squared_deviations = deviations ** 2  # [m, d]

        # 计算每个客户端的内部方差 + 均值偏差的加权和
        weighted_internal_vars = variances_tensor * weights_tensor.unsqueeze(1)  # [m, d]
        weighted_squared_deviations = squared_deviations * weights_tensor.unsqueeze(1)  # [m, d]

        # 合并方差
        global_variance = (torch.sum(weighted_internal_vars, dim=0) +
                           torch.sum(weighted_squared_deviations, dim=0)) / total_weight

        # 确保方差为正
        global_variance = torch.clamp(global_variance, min=1e-6)

        # 存储结果
        global_protos[class_id] = global_mean
        global_vars[class_id] = global_variance

    return global_protos, global_vars