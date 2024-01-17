import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ML_training.vanilla_training.models.GNNs import EMG_GCN, EMG_GAT
from ML_training.vanilla_training.models.CNN import EMG_CNN
from ML_training.vanilla_training.data_loader import MyDataset_loss_CH
from torch_geometric.loader import DataLoader
import time
from barbar import Bar
import ML_training.vanilla_training.helper
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import argparse, sys
from torch.utils.data.sampler import SubsetRandomSampler
from gifs.gesture_master_list import GESTURE_KEYS
import wandb
# import torch.profiler

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# np.random.seed(5111785)
# torch.manual_seed(5111785)

def train(model, train_loader, criterion, optimizer, scaler,args):
    model.train()
    t_train = time.time()
    loss_epoch = 0
    preds_out = []
    labels_out = []
    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
        
    for i, inp_train in enumerate(Bar(train_loader), 0):
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            inputs_list, labels_train = inp_train[0], inp_train[1].to(device)
            batch_items = labels_train.size()[0]
            inputs = inputs_list[0].to(device)
            num_nodes = inputs_list[1]
            preds, _ = model(inputs,batch_items, num_nodes)
            labels_train= F.one_hot(labels_train, num_classes=args.num_classes)
            loss = criterion(preds,labels_train.to(dtype = torch.float))
        scaler.scale(loss.mean()).backward()
        scaler.step(optimizer)
        scaler.update()
        preds_out+=list(preds.argmax(dim=1).cpu().numpy().flatten())
        labels_out+=list(labels_train.argmax(dim=1).cpu().numpy().flatten())
        
        loss_epoch += loss.mean().item()
            # prof.step()
    # sched.step()
    print("\nTrain Loss:", np.float16(loss_epoch/len(train_loader)),", Train Epoch Time:", np.float16(time.time()- t_train))
    return loss_epoch/len(train_loader), np.array(preds_out).flatten(), np.array(labels_out).flatten()


def val(model, valid_loader,criterion,args):
    model.eval()
    loss_epoch = 0
    preds_out = []
    labels_out = []
    for i, inp_train in enumerate(valid_loader):
        
        inputs_list, labels_train = inp_train[0], inp_train[1].to(device)
        batch_items = labels_train.size()[0]
        inputs = inputs_list[0].to(device)
        num_nodes = inputs_list[1]
        preds, _ = model(inputs,batch_items, num_nodes)
        labels_train= F.one_hot(labels_train, num_classes=args.num_classes)
        loss = criterion(preds,labels_train.to(dtype = torch.float))

        preds_out+=list(preds.argmax(dim=1).cpu().numpy().flatten())
        labels_out+=list(labels_train.argmax(dim=1).cpu().numpy().flatten())

        loss_epoch += loss.mean().item()
    t_train = time.time()
    print("\nVal Loss:", np.float16(loss_epoch/len(valid_loader)))
    # print("Val Epoch Time:", time.time()- t_train)
    return loss_epoch/len(valid_loader), np.array(preds_out).flatten(), np.array(labels_out).flatten()


def run_train_loss_CH(args):
    wandb.init()
    if args.train_combined:
        dataset = MyDataset_loss_CH(args.data,args.data_ext,args.excl_participant,args.excl_class)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor((1-args.train_test_split) * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size,num_workers = 6,pin_memory = True,sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=args.batch_size,num_workers = 6,pin_memory = True,sampler=valid_sampler)
    else:
        dataset_train = MyDataset_loss_CH(args.data,args.data_ext,args.excl_participant,args.excl_class)
        dataset_test = MyDataset_loss_CH(args.data,args.data_ext,args.incl_participant,args.excl_class)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size,num_workers = 6,pin_memory = True,shuffle = True)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size,num_workers = 6,pin_memory = True)

    print("Num Batches: ",len(train_loader))

    if args.model == 'GCN':
        net = EMG_GCN(args)
    elif args.model == 'GAT':
        net = EMG_GAT(args)
    elif args.model == 'CNN':
        net = EMG_CNN(args)
    else:
        print('model not found')
        raise NotImplementedError
    # optim_fn = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    optim_fn = optim.Adam(net.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(optim_fn, factor=0.5, patience=1, verbose=True, threshold=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    net.to(device)
    print(net)

    accuracy = 0
    for epoch in range(args.num_epochs):
        print("Epoch: ", epoch+1)
        t_0 = time.time()
        train_loss_epoch, preds_out, labels_out = train(net, train_loader, crit, optim_fn,scaler, args)
        accuracy_tr = np.float16(accuracy_score(labels_out, preds_out))
        print("Train Accuracy: ",np.float16(accuracy_tr*100),"%")
        test_loss_epoch, preds_out, labels_out = val(net, test_loader,crit,args)
        accuracy_itr = np.float16(accuracy_score(labels_out, preds_out))
        print("Test Accuracy: ",np.float16(accuracy_itr*100),"%")

        
        wandb.log({
        'epoch': epoch, 
        'train_acc': accuracy_tr,
        'train_loss': train_loss_epoch, 
        'val_acc': accuracy_itr, 
        'val_loss': test_loss_epoch
        })

        if accuracy_itr > accuracy and accuracy_itr>args.model_save_th:
            PATH_model = args.model_save_dir+args.model+'_'+str(accuracy_itr*100)+'_'+str(epoch)+'.pth'
            checkpoint = {'epoch': epoch,
                        'model': net.state_dict(),
                        'optim_label': optim_fn.state_dict(),
                        'sched_label': sched.state_dict(),
                        'scaler': scaler.state_dict(),
                        'gesture_key': GESTURE_KEYS,
                        'args': args}
            torch.save(checkpoint, PATH_model)
            pred_dict = {"preds":preds_out, "true":labels_out}
            ML_training.vanilla_training.helper.pkl_dump(args.model_save_dir+args.model+'_'+str(accuracy_itr*100)+'_'+str(epoch)+'_preds.pkl',pred_dict)
            accuracy = accuracy_itr
        sched.step(test_loss_epoch)
        if optim_fn.param_groups[0]['lr'] < args.convergence_lr:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = argparse.ArgumentParser(description='EMG_GNN')
    parser.add_argument('--data_train', default="data/train_BP/",
                        help = "Direcotry with training graphs")
    parser.add_argument('--data_val', default="data/test_BP/",
                        help = "Direcotry with testing graphs")

    # Architecture and training params
    parser.add_argument('--input_feat_dim',default=150, type=int,
                        help= "feature dimension size for input")
    parser.add_argument('--num_GCN_layers',default=2, type=int,
                        help='number of GCN layers to use in model (default:4)')
    parser.add_argument('--num_GCNneurons',default=128, type=int,
                        help='list of number of neurons to use per GCN Layer (default:256)')
    parser.add_argument('--k_pooling_dim',default=64, type=int,
                        help= "size of k-pooling layer output dimension")
    parser.add_argument('--num_fc_layer',default=3, type=int,
                        help='number of FC to use  to generate predictions (default: 3)')
    parser.add_argument('--fc_layer_scaling',default=16, type=int,
                        help='Scaling of FC layers wrt graph output dimension (last layer has the output class dimension)')
    parser.add_argument('--num_classes',default=8, type=int,
                        help='number of label classes in data')
    parser.add_argument('--dropout_rate',default=0.5, type=float)
    parser.add_argument('--num-epochs',default=200, type=int)
    parser.add_argument('--batch-size',default=128, type=int)
    parser.add_argument('--lr',default=0.001, type=float)
    parser.add_argument('--convergence_lr',default=1e-6, type=float)
    
    # Run specific params
    parser.add_argument('--model_save_dir',default= "model_wt/")
    parser.add_argument('--model_save_th',default= 0.7, type = int,
                        help = "Accuracy threshold for saving a model")

    args = parser.parse_args(sys.argv[1:])