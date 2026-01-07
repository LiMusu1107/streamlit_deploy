from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize

def evaluate_model(y_true, y_pred, y_proba, class_num):
    """计算各类性能指标（支持二分类和多分类）
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（对于二分类应为正类的概率）
        class_num: 类别数量
        
    返回:
        包含各种评估指标的字典
    """
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Macro": f1_score(y_true, y_pred, average='macro'),
        "F1 Micro": f1_score(y_true, y_pred, average='micro'),
        "F1 Weighted": f1_score(y_true, y_pred, average='weighted'),
        "Precision Macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Precision Micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Precision Weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall Macro": recall_score(y_true, y_pred, average='macro'),
        "Recall Micro": recall_score(y_true, y_pred, average='micro'),
        "Recall Weighted": recall_score(y_true, y_pred, average='weighted'),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred)
    }

    # 二分类 AUC 和 AUPR 的特殊处理
    if class_num == 2:
        # y_proba 应为正类的概率（形状 (n_samples,)）
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]  # 取第二列（正类的概率）
        scores.update({
            "AUC Macro": roc_auc_score(y_true, y_proba),
            "AUC Micro": roc_auc_score(y_true, y_proba),
            "AUC Weighted": roc_auc_score(y_true, y_proba),
            "AUPR Macro": average_precision_score(y_true, y_proba),
            "AUPR Micro": average_precision_score(y_true, y_proba),
            "AUPR Weighted": average_precision_score(y_true, y_proba),
        })
    else:
        # 多分类处理
        y_true_binarized = label_binarize(y_true, classes=list(range(class_num)))
        scores.update({
            "AUC Macro": roc_auc_score(y_true_binarized, y_proba, multi_class='ovr', average='macro'),
            "AUC Micro": roc_auc_score(y_true_binarized, y_proba, multi_class='ovr', average='micro'),
            "AUC Weighted": roc_auc_score(y_true_binarized, y_proba, multi_class='ovr', average='weighted'),
            "AUPR Macro": average_precision_score(y_true_binarized, y_proba, average='macro'),
            "AUPR Micro": average_precision_score(y_true_binarized, y_proba, average='micro'),
            "AUPR Weighted": average_precision_score(y_true_binarized, y_proba, average='weighted'),
        })
    return scores


def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    else:
        print("CUDA not available")

def count_trainable_parameters(model_dict):
    """计算模型字典中所有模型的可训练参数总数"""
    total_params = 0
    for model_name, model in model_dict.items():
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params += model_params
    return total_params

def get_model_info(model_dict):
    """获取详细的模型参数信息"""
    total_params = 0
    model_info = {}
    
    for model_name, model in model_dict.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_model = sum(p.numel() for p in model.parameters())
        
        model_info[model_name] = {
            'trainable_params': trainable_params,
            'total_params': total_params_model
        }
        total_params += trainable_params
    
    return total_params, model_info