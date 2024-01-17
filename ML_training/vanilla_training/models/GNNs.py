import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, conv
from torch_geometric.utils import add_self_loops, degree

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
    
class EMG_GCN(nn.Module):

    def __init__(self, args):
        super(EMG_GCN, self).__init__()

        self.GCN_layers = nn.ModuleList([GCNConv(args.input_feat_dim,args.num_GCNneurons)])
        for layer in range(1,args.num_GCN_layers):
            self.GCN_layers.append(GCNConv(args.num_GCNneurons,args.num_GCNneurons))

        self.pool_size = args.k_pooling_dim
        self.graph_out_size = args.num_GCNneurons
        self.global_pool1 = aggr.SortAggregation(k = args.k_pooling_dim)

        self.FC_scale = int(self.pool_size*self.graph_out_size)
        self.FC_layers = nn.ModuleList([torch.nn.Linear(self.FC_scale,int(self.FC_scale/args.fc_layer_scaling)),nn.Dropout(p=args.dropout_rate)])
        for layer in range(1,args.num_fc_layer-1):
            self.FC_layers.append(torch.nn.Linear(int(self.FC_scale/args.fc_layer_scaling),int(self.FC_scale/args.fc_layer_scaling)))
            self.FC_layers.append(nn.Dropout(p=args.dropout_rate))
        self.FC_layers.append(torch.nn.Linear(int(self.FC_scale/args.fc_layer_scaling),args.num_classes))
        for l in self.FC_layers:
            if isinstance(l, nn.Linear):
                torch.nn.init.xavier_normal_(l.weight, gain=1.0)
        
    def forward(self, graph,batch_items,num_nodes):
        total_nodes, feats = graph.x.size()
        inp = graph.x.float()
        for i,l in enumerate(self.GCN_layers):
            inp = F.relu(l(inp, graph.edge_index, graph.edge_attr))
        prev_node = 0
        holder_pool = torch.zeros((batch_items,self.pool_size*self.graph_out_size)).to(device)
        for i,idx in enumerate(num_nodes):
            pool_sample = inp[prev_node:prev_node+idx]
            pool1 = self.global_pool1(pool_sample)
            holder_pool[i,:] = pool1
            prev_node += idx
        
        for i in range(len(self.FC_layers)-1):
            holder_pool = F.relu(self.FC_layers[i](holder_pool))
        return F.softmax(self.FC_layers[-1](holder_pool), dim=1), holder_pool

