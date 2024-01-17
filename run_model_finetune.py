import sys, argparse, os
import torch
import ML_training.SSL_training.finetune
from gifs.gesture_master_list import GESTURE_KEYS

if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(5111785)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--data', default="./ML_training/training_data/GNN_samples/offset_1100_envelope_False/",
                        help = "Directory with training graphs")
    parser.add_argument('--data_ext', default=".pkl",
                        help = "file ext of samples")
    parser.add_argument('--train_combined', default=False,
                        help = "Train on all participants")
    parser.add_argument('--excl_participant', default=['P_3'],
                        help = "participant to test, eg. ['p1']")
    parser.add_argument('--incl_participant', default=['P_1','P_2'],
                        help = "participant to train, eg. ['p1']")
    parser.add_argument('--excl_class', default=[],
                        help = "class to exclude, eg. ['0']")
    parser.add_argument('--train_test_split', default=0.2,
                        help = "ratio of train test split")

    parser.add_argument('--num_fc_layer',default=2, type=int,
                        help='number of FC to use  to add to pretrained network (default: 3)')
    parser.add_argument('--fc_layer_scaling',default=1, type=int,
                        help='Scaling of FC layers wrt frozen model (last layer has the output class dimension)')

    parser.add_argument('--num_classes',default=len(GESTURE_KEYS), type=int,
                        help='number of label classes in data')
    parser.add_argument('--dropout_rate',default=0.5, type=float)
    parser.add_argument('--num_epochs',default=200, type=int)
    parser.add_argument('--batch-size',default=256, type=int)
    parser.add_argument('--lr',default=0.001, type=float)
    parser.add_argument('--convergence_lr',default=1e-6, type=float)

    
    parser.add_argument('--model_name', default="GCN_377.05330460159865_0",
                        help = "Model to be evaluated")
    parser.add_argument('--model_dir', default="./ML_training/SSL_training/model_wt",
                        help = "Directory with model")
    parser.add_argument('--model_save_dir',default= "./ML_training/vanilla_training/model_wt/")
    parser.add_argument('--model_save_th',default= 0.7, type = int,
                        help = "Accuracy threshold for saving a model")
    parser.add_argument('--save_dir', default="./ML_training/vanilla_training/model_wt",
                        help = "Directory to write model evals")

    args = parser.parse_args(sys.argv[1:])

    checkpoint = torch.load(os.path.join(args.model_dir,args.model_name+'.pth'), map_location=device)

    ML_training.SSL_training.finetune.finetune(checkpoint,args)
