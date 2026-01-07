""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from mogonet.models_mid_FNN import init_model_dict, init_optim
from mogonet.utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


# def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
#     loss_dict = {}
#     criterion = torch.nn.CrossEntropyLoss(reduction='none')
#     for m in model_dict:
#         model_dict[m].train()    
#     num_view = len(data_list)
#     for i in range(num_view):
#         optim_dict["C{:}".format(i+1)].zero_grad()
#         ci_loss = 0
#         ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
#         ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
#         ci_loss.backward()
#         optim_dict["C{:}".format(i+1)].step()
#         loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
#     if train_VCDN and num_view >= 2:
#         optim_dict["C"].zero_grad()
#         c_loss = 0
#         ci_list = []
#         for i in range(num_view):
#             ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
#         c = model_dict["C"](ci_list)    
#         c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
#         c_loss.backward()
#         optim_dict["C"].step()
#         loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
#     return loss_dict
    

# def test_epoch(data_list, adj_list, te_idx, model_dict):
#     for m in model_dict:
#         model_dict[m].eval()
#     num_view = len(data_list)
#     ci_list = []
#     for i in range(num_view):
#         ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
#     if num_view >= 2:
#         c = model_dict["C"](ci_list)    
#     else:
#         c = ci_list[0]
#     c = c[te_idx,:]
#     prob = F.softmax(c, dim=1).data.cpu().numpy()
    
#     return prob

# def train_epoch(data_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
#     loss_dict = {}
#     criterion = torch.nn.CrossEntropyLoss(reduction='none')
#     for m in model_dict:
#         model_dict[m].train()    
#     num_view = len(data_list)
#     for i in range(num_view):
#         optim_dict["C{:}".format(i+1)].zero_grad()
#         ci_loss = 0
#         ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i]))
#         ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
#         ci_loss.backward()
#         optim_dict["C{:}".format(i+1)].step()
#         loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
#     if train_VCDN and num_view >= 2:
#         optim_dict["C"].zero_grad()
#         c_loss = 0
#         ci_list = []
#         for i in range(num_view):
#             ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i])))
#         c = model_dict["C"](ci_list)    
#         c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
#         c_loss.backward()
#         optim_dict["C"].step()
#         loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
#     return loss_dict
    

# def test_epoch(data_list, te_idx, model_dict):
#     for m in model_dict:
#         model_dict[m].eval()
#     num_view = len(data_list)
#     ci_list = []
#     for i in range(num_view):
#         ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i])))
#     if num_view >= 2:
#         c = model_dict["C"](ci_list)    
#     else:
#         c = ci_list[0]
#     c = c[te_idx,:]
#     prob = F.softmax(c, dim=1).data.cpu().numpy()
    
#     return prob

def train_epoch(data_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_fusion=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # 设置所有模型为训练模式
    for m in model_dict:
        model_dict[m].train()
    
    num_view = len(data_list)
    
    if train_fusion and num_view >= 1:
        # 训练融合模型（编码器 + 分类器）
        optim_dict["E"].zero_grad()  # 清零编码器梯度
        optim_dict["C"].zero_grad()  # 清零分类器梯度
        
        # 获取所有视图的编码特征
        encoded_features = []
        for i in range(num_view):
            encoded_feat = model_dict["E{:}".format(i+1)](data_list[i])
            encoded_features.append(encoded_feat)
        
        # 特征concatenation
        concat_features = torch.cat(encoded_features, dim=1)
        
        # 融合分类
        fusion_output = model_dict["C"](concat_features)
        
        # 计算损失
        fusion_loss = torch.mean(torch.mul(criterion(fusion_output, label), sample_weight))
        
        # 反向传播
        fusion_loss.backward()
        
        # 更新参数
        optim_dict["E"].step()  # 更新所有编码器
        optim_dict["C"].step()  # 更新分类器
        
        loss_dict["fusion"] = fusion_loss.detach().cpu().numpy().item()
    
    return loss_dict


def test_epoch(data_list, te_idx, model_dict):
    # 设置所有模型为评估模式
    for m in model_dict:
        model_dict[m].eval()
    
    num_view = len(data_list)
    
    with torch.no_grad():  # 测试时不需要梯度
        # 获取所有视图的编码特征
        encoded_features = []
        for i in range(num_view):
            encoded_feat = model_dict["E{:}".format(i+1)](data_list[i])
            encoded_features.append(encoded_feat)
        
        if num_view >= 2:
            # 多视图：特征concatenation + 融合分类
            concat_features = torch.cat(encoded_features, dim=1)
            c = model_dict["C"](concat_features)
        else:
            # 单视图：直接分类（需要确保有对应的单视图分类器）
            c = model_dict["C"](encoded_features[0])
    
    # 获取测试索引的结果
    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


# # 可选：如果你还想保留单独训练每个视图的功能，可以添加这个函数
# def train_epoch_individual(data_list, label, one_hot_label, sample_weight, model_dict, optim_dict):
#     """
#     单独训练每个视图（如果需要的话）
#     注意：这需要为每个视图创建单独的分类器
#     """
#     loss_dict = {}
#     criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
#     for m in model_dict:
#         model_dict[m].train()
    
#     num_view = len(data_list)
    
#     # 这种方式需要为每个视图创建单独的分类器 C1, C2, C3...
#     # 目前的代码结构只有一个融合分类器 C
#     for i in range(num_view):
#         if "C{:}".format(i+1) in model_dict:
#             # 如果存在单独的分类器
#             optim_individual = torch.optim.Adam(
#                 list(model_dict["E{:}".format(i+1)].parameters()) + 
#                 list(model_dict["C{:}".format(i+1)].parameters())
#             )
            
#             optim_individual.zero_grad()
            
#             # 编码 + 分类
#             encoded_feat = model_dict["E{:}".format(i+1)](data_list[i])
#             ci = model_dict["C{:}".format(i+1)](encoded_feat)
            
#             # 计算损失
#             ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
            
#             # 反向传播和更新
#             ci_loss.backward()
#             optim_individual.step()
            
#             loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    
#     return loss_dict





def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch):
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    adj_parameter = 3
    dim_he_list = [300, 100]

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    test_inverval = 5
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            print()