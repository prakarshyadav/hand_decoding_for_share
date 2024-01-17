import argparse, sys, os, glob
import numpy as np
from scipy import signal
from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_readers import Poly5Reader
import ML_training.graph_gen.helper 
import posixpath

def read_poly(fname):
    data = Poly5Reader(fname)
    samples = data.samples
    return samples

def align_trigs(trig_1,trig_2):
    correlation = signal.correlate(trig_1, trig_2, mode="full")
    lags = signal.correlation_lags(len(trig_1), len(trig_2), mode="full")
    lag = lags[np.argmax(abs(correlation))]
    if lag>0:
        align_idx_1 = lag
        align_idx_2 = 0
    elif lag < 0:
        align_idx_1 = 0
        align_idx_2 = lag
    else:
        align_idx_1 = 0
        align_idx_2 = 0
    return align_idx_1, align_idx_2

def remove_outliers(data,args):
    # import matplotlib.pyplot as plt

    zero_arr = np.zeros((data.shape[0],args.blanking_region))
    data[:,:args.blanking_region]= zero_arr
    data[:,-args.blanking_region:]= zero_arr

    outliers_all = np.empty(0,dtype = int)
    for chan in range(data.shape[0]):
        data_chan = data[chan,:]
        stdev = np.std(data_chan)*args.std_scaling
        outliers = np.where(np.abs(data_chan) > stdev)[0]
        if len(outliers)>args.outlier_th:
            data[chan,:] = np.zeros_like(data_chan)
        else:
            outliers_all = np. concatenate((outliers_all,outliers))
    data = np.delete(data,np.unique(outliers_all),axis = 1)
    # fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
    # counter = 0
    # for i in range(8):
    #     for j in range(8):
    #         axes[i][j].axhline(y=stdev, color='k')
    #         axes[i][j].axhline(y=-stdev, color='k')
    #         axes[i][j].plot(data[counter],alpha = 0.85)
    #         # axes[i][j].set_xlim([-LLIM,ULIM])
    #         # axes[i][j].set_ylim([0,150])
    #         axes[i][j].spines["right"].set_visible(False)
    #         axes[i][j].spines["top"].set_visible(False)
    #         # axes[i][j].spines["left"].set_visible(False)
    #         axes[i][j].spines["bottom"].set_visible(False)
    #         axes[i][j].get_xaxis().set_visible(False) # Hide tick marks and spines
    #         # axes[i][j].get_yaxis().set_visible(False)
    #         counter +=1
    # axes[-1][0].spines["left"].set_visible(True)
    # axes[-1][0].spines["bottom"].set_visible(True)
    # axes[-1][0].get_xaxis().set_visible(True) 
    # axes[-1][0].get_yaxis().set_visible(True)
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    return data

def process_poly5(dir_name,args):
    file_id = []
    for fname in sorted(glob.glob(dir_name+'/*.poly5')):
        file_id.append(fname)
    
    if len(file_id)%2 != 0:
        print("something weird with this trial ",dir_name)
        raise FileNotFoundError
    data_dir = {}
    for i in range(len(file_id)//2):
        fname_1=file_id[i*2].replace(os.sep, posixpath.sep)
        fname_2=file_id[2*i+1].replace(os.sep, posixpath.sep)
        if fname_1.split('/')[-1].split('_')[0] != fname_2.split('/')[-1].split('_')[0]:
            print("Files are not matched")
            raise NotImplementedError
        else:
            poly5_1 = read_poly(fname_1)
            f1_trigs = poly5_1[-3,:]
            f1_samps = poly5_1[:64,:]
            poly5_2 = read_poly(fname_2)
            f2_trigs = poly5_2[-3,:]
            f2_samps = poly5_2[:64,:]
            idx_1, idx_2 = align_trigs(f1_trigs,f2_trigs)
            f1_trigs_trunc = f1_trigs[idx_1:]
            f1_samps_trunc = f1_samps[:,idx_1:]
            f2_trigs_trunc = f2_trigs[idx_2:]
            f2_samps_trunc = f2_samps[:,idx_2:]
            min_len = min(f1_trigs_trunc.shape[0],f2_trigs_trunc.shape[0])
            f1_trigs_trunc = 1-f1_trigs_trunc[:min_len]
            f1_samps_trunc = f1_samps_trunc[:,:min_len]
            f2_trigs_trunc = 1-f2_trigs_trunc[:min_len]
            f2_samps_trunc = f2_samps_trunc[:,:min_len]
            f1_clean = remove_outliers(ML_training.graph_gen.helper.filt_GRID(f1_samps_trunc),args)
            f2_clean = remove_outliers(ML_training.graph_gen.helper.filt_GRID(f2_samps_trunc),args)
            data_bloc = {
                'trigs': np.array([f1_trigs_trunc,f2_trigs_trunc]),
                'samp1': f1_clean,
                'samp2': f2_clean,
            }
            print(fname_1.split('/')[-1].split('_')[0])
            data_dir[fname_1.split('/')[-1].split('_')[0]] = data_bloc
    return data_dir

def convert_poly5_2_pkl(args):
    participants = os.listdir(args.data_dir)
    for participant in participants:
        if participant.split('_')[-1] not in ['eval']:
            exp_dates = os.listdir(os.path.join(args.data_dir,participant))
            for exp_date in exp_dates:
                pkl_save_path = os.path.join(args.data_dir,participant,exp_date)+'/' +exp_date+'.pkl'
                if not os.path.isfile(pkl_save_path):
                    print("Running: ", participant,exp_date)
                    poses = os.listdir(os.path.join(args.data_dir,participant,exp_date))
                    pose_dict = {}
                    for pose in poses:
                        gestures = os.listdir(os.path.join(args.data_dir,participant,exp_date,pose))
                        gest_dict = {}
                        for gesture in gestures:
                            gesture_path = os.path.join(args.data_dir,participant,exp_date,pose,gesture)
                            gest_dict[gesture] = process_poly5(gesture_path,args,)
                        pose_dict[pose] = gest_dict
                    ML_training.graph_gen.helper.pkl_dump(pkl_save_path, pose_dict)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description='Poly5_2_pkl')
    parser.add_argument('--data_dir',default='data', type=str,
                        help= "Data directory")
    
    args = parser.parse_args(sys.argv[1:])
    convert_poly5_2_pkl(args)