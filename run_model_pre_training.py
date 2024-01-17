import sys, argparse
# import ML_training.vanilla_training.train_model
import ML_training.SSL_training.pre_train_BT
import ML_training.SSL_training.pre_train_triplet
import torch, os
from gifs.gesture_master_list import GESTURE_KEYS

if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(5111785)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', default="./ML_training/training_data/GNN_samples/offset_1100_envelope_True/",
    # parser.add_argument('--data', default="/home/nml2080ti/Desktop/Gestures/GNN_samples/offset_1100_envelope_True/",
                        help = "Directory with training graphs")
    parser.add_argument('--data_ext', default=".pkl",
                        help = "file ext of samples")
    parser.add_argument('--train_combined', default=True,
                        help = "Train on all participants")
    parser.add_argument('--excl_participant', default=[],
                        help = "participant to test, eg. ['p1']")
    parser.add_argument('--incl_participant', default=['P_1','P_2','P_3','P_4','P_5','P_6'],
                        help = "participant to train, eg. ['p1']")
    parser.add_argument('--excl_class', default=[],
                        help = "class to exclude, eg. ['0']")
    parser.add_argument('--train_test_split', default=0.2,
                        help = "ratio of train test split")
    # Architecture and training params
    parser.add_argument('--model',default='GCN', type=str,
                        help= "model type to test")
    temp_check = parser.parse_args(sys.argv[1:])

    if temp_check.model =='GAT':
        parser.add_argument('--input_feat_dim',default=100, type=int,
                            help= "feature dimension size for input")
        parser.add_argument('--num_heads',default=2, type=int,
                            help='number of GAT heads')
        parser.add_argument('--num_GCN_layers',default=3, type=int,
                            help='number of GAT layers to use in model (default:4)')
        parser.add_argument('--num_GCNneurons',default=128, type=int,
                            help='list of number of neurons to use per GCN Layer (default:256)')
        parser.add_argument('--k_pooling_dim',default=64, type=int,
                            help= "size of k-pooling layer output dimension") 
        parser.add_argument('--num_fc_layer',default=4, type=int,
                            help='number of FC to use  to generate predictions (default: 3)')
        parser.add_argument('--fc_layer_scaling',default=8, type=int,
                            help='Scaling of FC layers wrt graph output dimension (last layer has the output class dimension)')

    elif temp_check.model =='GCN':
        parser.add_argument('--input_feat_dim',default=100, type=int,
                            help= "feature dimension size for input")
        parser.add_argument('--num_GCN_layers',default=3, type=int,
                            help='number of GCN layers to use in model (default:4)')
        parser.add_argument('--num_GCNneurons',default=256, type=int,
                            help='list of number of neurons to use per GCN Layer (default:256)')
        parser.add_argument('--k_pooling_dim',default=16, type=int,
                            help= "size of k-pooling layer output dimension")
        parser.add_argument('--num_fc_layer',default=3, type=int,
                            help='number of FC to use  to generate predictions (default: 3)')
        parser.add_argument('--fc_layer_scaling',default=4, type=int,
                            help='Scaling of FC layers wrt graph output dimension (last layer has the output class dimension)')
        
    elif temp_check.model =='CNN':
        parser.add_argument('--input_feat_dim',default=100, type=int,
                            help= "feature dimension size for input")
        parser.add_argument('--out_chan_scale',default=1, type=int,
                            help= "output feature dimension scale factor")
        parser.add_argument('--input_size_dim',default=(8,16), type=tuple,
                            help= "feature dimension size for input (w,h)")
        parser.add_argument('--num_CNN_layers',default=3, type=int,
                            help='number of CNN layers to use in model (default:4)')
        parser.add_argument('--num_CNNkernels',default=3, type=int,
                            help='list of number of neurons to use per GCN Layer (default:256)')
        parser.add_argument('--max_pool_size',default=2, type=int,
                            help= "size of max-pooling layer output dimension")
        parser.add_argument('--num_fc_layer',default=1, type=int,
                            help='number of FC to use  to generate predictions (default: 3)')
        parser.add_argument('--fc_layer_scaling',default=2, type=int,
                            help='Scaling of FC layers wrt graph output dimension (last layer has the output class dimension)')
    else:
        print("model selected is not yet implemented")
        raise NotImplementedError

    parser.add_argument('--num_classes',default=len(GESTURE_KEYS), type=int,
                        help='number of label classes in data')
    parser.add_argument('--dropout_rate',default=0.5, type=float)
    parser.add_argument('--num_epochs',default=200, type=int)
    parser.add_argument('--batch-size',default=256, type=int)
    parser.add_argument('--lr',default=0.001, type=float)
    parser.add_argument('--convergence_lr',default=1e-5, type=float)
    
    # triplet args

    parser.add_argument('--margin', default=0.5, type=float, 
                        help='margin for Triplet loss')
    parser.add_argument('--triplet_scaling', default=0.15, type=float, 
                        help='scaling factor for Triplet loss')
    # BT args

    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--projector_layers', default=3, type=int,
                        help='projector MLP')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--vis_freq', default=10, type=int, metavar='V',
                        help='print frequency')

    # Run specific params

    parser.add_argument('--model_name', default="GCN_80.517578125_32",
                        help = "Model to be evaluated")
    parser.add_argument('--model_dir', default="./ML_training/vanilla_training/model_wt",
                        help = "Directory with model")
    parser.add_argument('--model_save_dir',default= "./ML_training/SSL_training/model_wt/")
    parser.add_argument('--model_save_th',default= 0.7, type = int,
                        help = "Accuracy threshold for saving a model")

    args = parser.parse_args(sys.argv[1:])

    checkpoint = torch.load(os.path.join(args.model_dir,args.model_name+'.pth'), map_location=device)

    # ML_training.SSL_training.pre_train_BT.run_pretrain_BT(args)
    ML_training.SSL_training.pre_train_triplet.run_pretrain_triplet(checkpoint, args)