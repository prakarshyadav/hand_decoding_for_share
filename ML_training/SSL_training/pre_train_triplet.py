import ML_training.SSL_training.pre_train_utils
import ML_training.SSL_training.model.make_BT
import torch
import torch.nn as nn
import torch.optim as optim
from barbar import Bar
from torch.utils.data.sampler import SubsetRandomSampler
import time, json
from ML_training.SSL_training.data_loader import MyDataset_loss_paired, MyDataset_loss_triplet
from torch_geometric.loader import DataLoader
from ML_training.vanilla_training.models.GNNs import EMG_GCN, EMG_GAT
from ML_training.vanilla_training.models.CNN import EMG_CNN
import ML_training.vanilla_training.helper
from gifs.gesture_master_list import GESTURE_KEYS
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import ML_training.SSL_training.pre_train_utils
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity,LpDistance
from pytorch_metric_learning.reducers import ThresholdReducer
import ML_training.SSL_training.pre_train_utils
from pytorch_metric_learning.regularizers import LpRegularizer
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def train(model, train_loader, criterion, optimizer, scaler, miner,args):
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
            anchor_inputs, positive_inputs, negative_inputs, num_nodes,labels_train, label_neg = inp_train[0].to(device), inp_train[1].to(device), inp_train[2].to(device), inp_train[3].to(device),inp_train[4].to(device), inp_train[5].to(device)
            # inputs_list, labels_train = inp_train[0], inp_train[1].to(device)
            batch_items = labels_train.size()[0]
            pair_labels = torch.cat((labels_train,labels_train,label_neg),0)
            # inputs = inputs_list[0].to(device)
            # num_nodes = inputs_list[1]
            preds, anchor = model(anchor_inputs,batch_items, num_nodes)
            _, positive = model(positive_inputs,batch_items, num_nodes)
            _, negative = model(negative_inputs,batch_items, num_nodes)
            labels_train= F.one_hot(labels_train, num_classes=args.num_classes)
            embeds = torch.cat((anchor,positive,negative),0)
            hard_pairs = miner(embeds, pair_labels)
            loss = criterion(embeds, pair_labels, hard_pairs)
            # loss = criterion(anchor, positive, negative)
            # loss = criterion(preds,labels_train.to(dtype = torch.float))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        preds_out+=list(preds.argmax(dim=1).cpu().numpy().flatten())
        labels_out+=list(labels_train.argmax(dim=1).cpu().numpy().flatten())
        
        loss_epoch += loss.mean().item()
            # prof.step()
    print("\nTrain Loss:", np.float16(loss_epoch/len(train_loader)),", Train Epoch Time:", np.float16(time.time()- t_train))
    return loss_epoch/len(train_loader), np.array(preds_out).flatten(), np.array(labels_out).flatten()


def val(model, valid_loader,criterion,miner, args):
    model.eval()
    loss_epoch = 0
    preds_out = []
    labels_out = []
    for i, inp_train in enumerate(Bar(valid_loader),0):
        anchor_inputs, positive_inputs, negative_inputs, num_nodes,labels_train, label_neg = inp_train[0].to(device), inp_train[1].to(device), inp_train[2].to(device), inp_train[3].to(device),inp_train[4].to(device), inp_train[5].to(device)
        # inputs_list, labels_train = inp_train[0], inp_train[1].to(device)
        batch_items = labels_train.size()[0]
        pair_labels = torch.cat((labels_train,labels_train,label_neg),0)
        # inputs = inputs_list[0].to(device)
        # num_nodes = inputs_list[1]
        preds, anchor = model(anchor_inputs,batch_items, num_nodes)
        _, positive = model(positive_inputs,batch_items, num_nodes)
        _, negative = model(negative_inputs,batch_items, num_nodes)
        labels_train= F.one_hot(labels_train, num_classes=args.num_classes)

        embeds = torch.cat((anchor,positive,negative),0)
        hard_pairs = miner(embeds, pair_labels)
        loss = criterion(embeds, pair_labels, hard_pairs)
        # loss = criterion(anchor, positive, negative)
        
        # inputs_list, labels_train = inp_train[0], inp_train[1].to(device)
        # batch_items = labels_train.size()[0]
        # inputs = inputs_list[0].to(device)
        # num_nodes = inputs_list[1]
        # preds, _ = model(inputs,batch_items, num_nodes)
        # labels_train= F.one_hot(labels_train, num_classes=args.num_classes)
        # loss = criterion(preds,labels_train.to(dtype = torch.float))

        preds_out+=list(preds.argmax(dim=1).cpu().numpy().flatten())
        labels_out+=list(labels_train.argmax(dim=1).cpu().numpy().flatten())

        loss_epoch += loss.mean().item()
    t_train = time.time()
    print("\nVal Loss:", np.float16(loss_epoch/len(valid_loader)))
    # print("Val Epoch Time:", time.time()- t_train)
    return loss_epoch/len(valid_loader), np.array(preds_out).flatten(), np.array(labels_out).flatten()



def run_pretrain_triplet(chkp, args):
    train_args = chkp['args']
    # wandb.init()
    if args.train_combined:
        dataset = MyDataset_loss_triplet(args.data,args.data_ext,args.excl_participant,args.excl_class)
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
        dataset_train = MyDataset_loss_triplet(args.data,args.data_ext,args.excl_participant,args.excl_class)
        dataset_test = MyDataset_loss_triplet(args.data,args.data_ext,args.incl_participant,args.excl_class)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size,num_workers = 6,pin_memory = True,shuffle = True)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size,num_workers = 6,pin_memory = True)

        # dataset_train = MyDataset_loss_triplet(args.data,args.data_ext,args.excl_participant,args.excl_class)
        # train_loader = DataLoader(dataset_train, batch_size=args.batch_size,num_workers = 6,pin_memory = True,shuffle = True)
        # dataset_test = MyDataset_loss_paired(args.data,args.data_ext,args.incl_participant,args.excl_class)
        # test_loader = DataLoader(dataset_test, batch_size=args.batch_size,num_workers = 6,pin_memory = True)

    print("Num Batches: ",len(train_loader))

    if train_args.model == 'GCN':
        net = EMG_GCN(train_args)
    elif train_args.model == 'GAT':
        net = EMG_GAT(train_args)
    elif train_args.model == 'CNN':
        net = EMG_CNN(train_args)
    else:
        print('model not found')
        raise NotImplementedError
    # net = ML_training.SSL_training.pre_train_utils.match_wts(net, chkp['model'], args)
    # optim_fn = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.load_state_dict(chkp['model'])

    # for param in net.parameters():
    #     param.requires_grad = False

    optim_fn = optim.Adam(net.parameters(), lr=args.lr)
    # crit = nn.CrossEntropyLoss()

    miner = miners.TripletMarginMiner(margin=0.3, type_of_triplets="semihard")
    crit = losses.TripletMarginLoss(distance = LpDistance(), 
				     reducer = ThresholdReducer(high=0.3), 
			 	     embedding_regularizer = LpRegularizer())
    # crit = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=args.margin)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optim_fn, factor=0.5, patience=2, verbose=True, threshold=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    net.to(device)
    print(net)

    loss_th = 1e15
    accuracy = 0
    for epoch in range(args.num_epochs):
        print("Epoch: ", epoch+1)
        t_0 = time.time()
        train_loss_epoch, preds_out, labels_out = train(net, train_loader, crit, optim_fn, scaler, miner, args)
        accuracy_tr = np.float16(accuracy_score(labels_out, preds_out))
        print("Train Accuracy: ",np.float16(accuracy_tr*100),"%")
        test_loss_epoch, preds_out, labels_out = val(net, test_loader,crit,miner, args)
        accuracy_itr = np.float16(accuracy_score(labels_out, preds_out))
        print("Test Accuracy: ",np.float16(accuracy_itr*100),"%")

        
        # wandb.log({
        # 'epoch': epoch, 
        # 'train_acc': accuracy_tr,
        # 'train_loss': train_loss_epoch, 
        # 'val_acc': accuracy_itr, 
        # 'val_loss': test_loss_epoch
        # })

        if test_loss_epoch < loss_th:
            PATH_model = args.model_save_dir+args.model+'_'+str(train_loss_epoch)+'_'+str(epoch)+'.pth'
            checkpoint = {'epoch': epoch,
                        'model': net.state_dict(),
                        'optim_label': optim_fn.state_dict(),
                        'sched_label': sched.state_dict(),
                        'scaler': scaler.state_dict(),
                        'gesture_key': GESTURE_KEYS,
                        'args': args}
            torch.save(checkpoint, PATH_model)
            pred_dict = {"preds":preds_out, "true":labels_out}
            ML_training.vanilla_training.helper.pkl_dump(args.model_save_dir+args.model+'_'+str(train_loss_epoch)+'_'+str(epoch)+'_preds.pkl',pred_dict)
            accuracy = accuracy_itr
            loss_th = test_loss_epoch
        sched.step(test_loss_epoch)
        if optim_fn.param_groups[0]['lr'] < args.convergence_lr:
            break
