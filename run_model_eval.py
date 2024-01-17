import sys, argparse, os
import torch
import ML_training.vanilla_training.train_model
import ML_training.analysis.tsne
import ML_training.analysis.confusion_mat

if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(5111785)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_name', default="GCN_80.517578125_32",
                        help = "Model to be evaluated")
    parser.add_argument('--model_dir', default="./ML_training/vanilla_training/model_wt",
                        help = "Directory with model")
    parser.add_argument('--save_dir', default="./ML_training/vanilla_training/model_wt",
                        help = "Directory to write model evals")

    parser.add_argument('--data', default="./ML_training/training_data/GNN_samples/offset_1100_envelope_True/",
                        help = "Directory with training graphs")
    parser.add_argument('--data_ext', default=".pkl",
                        help = "file ext of samples")
    parser.add_argument('--excl_participant', default=['P_1','P_2','P_3','P_4','P_5'],
                        help = "participant to test, eg. ['p1']")
    parser.add_argument('--incl_participant', default=['P_6'],
                        help = "participant to train, eg. ['p1']")
    parser.add_argument('--excl_class', default=[],
                        help = "class to exclude, eg. ['0']")
    
    args = parser.parse_args(sys.argv[1:])

    checkpoint = torch.load(os.path.join(args.model_dir,args.model_name+'.pth'), map_location=device)
    if not os.path.exists(os.path.join(args.model_dir,args.model_name)): os.mkdir(os.path.join(args.model_dir,args.model_name))
    ML_training.analysis.confusion_mat.get_confusion_mat(checkpoint,args)
    ML_training.analysis.tsne.get_tsne(checkpoint,args)
