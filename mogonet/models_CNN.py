""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



# class NN_E(nn.Module):
#     def __init__(self, in_dim, hgcn_dim, dropout):
#         super().__init__()
#         # self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
#         # self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
#         # self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
#         self.gc1 = nn.Linear(in_dim, hgcn_dim[0])
#         self.gc2 = nn.Linear(hgcn_dim[0], hgcn_dim[1])
#         self.gc3 = nn.Linear(hgcn_dim[1], hgcn_dim[2])
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = self.gc1(x, adj)
#         x = F.leaky_relu(x, 0.25)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         x = F.leaky_relu(x, 0.25)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc3(x, adj)
#         x = F.leaky_relu(x, 0.25)
        
#         return x

# class NN_E(nn.Module):
#     def __init__(self, in_dim, hgcn_dim, dropout):
#         super().__init__()

#         self.fc1 = nn.Linear(in_dim, hgcn_dim[0])
#         self.fc2 = nn.Linear(hgcn_dim[0], hgcn_dim[1])
#         self.fc3 = nn.Linear(hgcn_dim[1], hgcn_dim[2])
#         self.dropout = dropout

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.leaky_relu(x, 0.25)
#         x = F.dropout(x, self.dropout, training=self.training)
        
#         x = self.fc2(x)
#         x = F.leaky_relu(x, 0.25)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.fc3(x)
#         x = F.leaky_relu(x, 0.25)
        
#         return x

class CNN1D_E(nn.Module):
    def __init__(self, in_dim, hid_dim, kernal_shape, maxpool_shape, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(1, hid_dim[0], kernal_shape[0])
        self.maxpool1 = nn.MaxPool1d(maxpool_shape[0])
        
        self.conv2 = nn.Conv1d(hid_dim[0], hid_dim[1], kernal_shape[1])
        self.maxpool2 = nn.MaxPool1d(maxpool_shape[1])
        
        self.in_dim = in_dim
        self.dropout_rate = dropout  # 存储dropout比率
        self._calculate_conv_output_size()
        
        self.fc1 = nn.Linear(self.conv_output_size, hid_dim[-1])

    def _calculate_conv_output_size(self):
        with torch.no_grad():
            x = torch.randn(1, 1, self.in_dim)
            x = F.relu(self.conv1(x))
            x = self.maxpool1(x)
            x = F.relu(self.conv2(x))
            x = self.maxpool2(x)
            self.conv_output_size = x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)
        
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        
        # 展平
        x = torch.flatten(x, start_dim=1)
        
        # 全连接层
        x = F.leaky_relu(self.fc1(x), 0.25)
        x = F.dropout(x, self.dropout_rate, training=self.training)  
        
        return x





class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)
        
    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
        for i in range(2,num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
        vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
        output = self.model(vcdn_feat)

        return output



    
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, kernal_dim, maxpool_dim, dropout_rate=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = CNN1D_E(dim_list[i], dim_he_list, kernal_dim, maxpool_dim,  dropout_rate)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict