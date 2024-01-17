from ML_training.vanilla_training.models.GNNs import EMG_GCN, EMG_GAT
from ML_training.vanilla_training.models.CNN import EMG_CNN
import ML_training.vanilla_training.helper
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
import torch
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import time
import matplotlib.pyplot as plt
import h5py
import os

def background_inference(model_pth, T0, event_start, event_term,event_align, MVC_arr, inp_dim):
    name = "decoder"
    n_channels = 1
    srate = 20
    chunk_size = inp_dim

    SAMPLING_TOL = 1e-5
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/ nyq, highcut/ nyq], btype='band', analog=False)
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y


    def filt_GRID(data, lowcut, highcut, fs, order=3, notch_fs = 50, notch_q = 30):
        filt_out = np.zeros_like(data)
        for i in range(data.shape[1]):
            filt_out[:,i] = notch_filter(butter_bandpass_filter(data[:,i], lowcut, highcut, fs, order=order), notch_fs, fs, notch_q)
        return filt_out

    def notch(notch_freq, samp_freq, quality_factor=30):
        b, a = iirnotch(notch_freq, quality_factor, samp_freq)
        return b, a

    def notch_filter(data, notch_fs, fs, q=30):
        b, a = notch(notch_fs, fs, q)
        y = filtfilt(b, a, data)
        return y

    ECR_MVC = MVC_arr["ECR"]
    FCR_MVC = MVC_arr["FCR"]
    MVC_vals = np.concatenate((ECR_MVC,FCR_MVC)).reshape(1,-1)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    CHKPT = torch.load(model_pth) 
    
    if CHKPT['args'].model == 'GCN':
        net = EMG_GCN(CHKPT['args'])
    elif CHKPT['args'].model == 'GAT':
        net = EMG_GAT(CHKPT['args'])
    elif CHKPT['args'].model == 'CNN':
        net = EMG_CNN(CHKPT['args'])
    else:
        print('model not found')
        raise NotImplementedError
    
    net.load_state_dict(CHKPT['model'])
    net.to(device)
    net.eval()
    
    print("finding stream")
    stream_ECR = resolve_stream('name', 'FCR')
    for info in stream_ECR:
        print('name: ', info.name())
        print('channel count:', info.channel_count())
        print('sampling rate:', info.nominal_srate())
        print('type: ', info.type())
    stream_FCR = resolve_stream('name', 'ECR')
    for info in stream_FCR:
        print('name: ', info.name())
        print('channel count:', info.channel_count())
        print('sampling rate:', info.nominal_srate())
        print('type: ', info.type())

    inlet_ECR = StreamInlet(stream_ECR[0])
    inlet_FCR = StreamInlet(stream_FCR[0])    
    print("pulling sample")
    sample_1, timestamp_1 = inlet_ECR.pull_sample()
    sample_2, timestamp_2 = inlet_FCR.pull_sample()
    timestamp_1 += inlet_ECR.time_correction()
    timestamp_2 += inlet_FCR.time_correction()
    print("trying coarse time alignment")
    while abs(timestamp_1 - timestamp_2) > SAMPLING_TOL:
        if timestamp_1 - timestamp_2 > SAMPLING_TOL:
            while timestamp_1 - timestamp_2 > SAMPLING_TOL:
                sample_2, timestamp_2 = inlet_FCR.pull_sample()
        elif timestamp_2 - timestamp_1  > SAMPLING_TOL:
            while timestamp_2 - timestamp_1 > SAMPLING_TOL:
                sample_1, timestamp_1 = inlet_ECR.pull_sample()
    sample_1 = np.array(sample_1).reshape(-1)
    sample_2 = np.array(sample_2).reshape(-1)
    event_start.set()
    print("sending signal for alignment")
    """
    Sampling loop to fix trigs, 10000 samples
    """
    samples_to_fix = 10000
    trig_1 = np.empty((samples_to_fix))
    trig_2 = np.empty((samples_to_fix))
    for i in range(samples_to_fix):
        sample_1, timestamp_1 = inlet_ECR.pull_sample()
        sample_2, timestamp_2 = inlet_FCR.pull_sample()
        trig_1[i] = sample_1[-3]#np.array(sample_1).reshape(-1)
        trig_2[i] = sample_2[-3]#np.array(sample_2).reshape(-1)
    # plt.plot(trig_1)
    # plt.plot(trig_2)
    # plt.savefig('align_in.png')
    # plt.close()
    flag, gap = align_trig(trig_1, trig_2)
    if flag == 0:
        for i in range(gap):
            sample_2, timestamp_2 = inlet_FCR.pull_sample()
    if flag == 1:
        for i in range(gap):
            sample_1, timestamp_1 = inlet_ECR.pull_sample()

    trig_1 = np.empty((samples_to_fix))
    trig_2 = np.empty((samples_to_fix))
    for i in range(samples_to_fix):
        sample_1, timestamp_1 = inlet_ECR.pull_sample()
        sample_2, timestamp_2 = inlet_FCR.pull_sample()
        trig_1[i] = sample_1[-3]#np.array(sample_1).reshape(-1)
        trig_2[i] = sample_2[-3]#np.array(sample_2).reshape(-1)
    # plt.plot(trig_1)
    # plt.plot(trig_2)
    # plt.savefig('align_out.png')
    # plt.close()

    sample_1 = np.array(sample_1).reshape(-1)
    sample_2 = np.array(sample_2).reshape(-1)

    print("stream Time", time.time()-T0)
    holder_1 = np.empty((chunk_size,sample_1.shape[0],))
    holder_2 = np.empty((chunk_size,sample_2.shape[0],))
    for i in range(chunk_size):
        sample_1, timestamp_1 = inlet_ECR.pull_sample()
        sample_2, timestamp_2 = inlet_FCR.pull_sample()
        sample_1 = np.array(sample_1).reshape(-1)
        sample_2 = np.array(sample_2).reshape(-1)
        holder_1[i,:] = sample_1
        holder_2[i,:] = sample_2

    
    event_align.set()
    
    crds = ML_training.vanilla_training.helper.pkl_load("./ML_training/graph_gen/crd_info_2grid_ECR_FCR.pkl")
    spacing,neigh_dist = 8.75, 12
    crds_scaled = crds * spacing
    edge_idx, edge_feat = ML_training.vanilla_training.helper.get_neigh_2D(crds_scaled, neigh_dist)
    array_data = np.concatenate((holder_1[:,0:64],holder_2[:,0:64]),axis = 1)

    dead_chan = np.where(np.nan_to_num(np.mean(array_data,axis = 0),nan=0,posinf=0,neginf=0) ==0)[0]
    for idx in dead_chan:
        array_data[:,idx] = np.zeros(array_data.shape[0])
    array_data_filt = filt_GRID(array_data,20,500,2000,)
    array_data_scaled = array_data_filt/MVC_vals 
    graph_instance = ML_training.vanilla_training.helper.generate_grid_graph_2D_infer(array_data_scaled, crds_scaled, edge_idx, edge_feat, dead_chan = dead_chan)
    batch_items = 1
    inputs = graph_instance.to(device)
    num_nodes = [graph_instance.num_nodes]
    preds,_ = net(inputs,batch_items, num_nodes)
    decoded_gesture = preds.argmax(dim=1).cpu().numpy().flatten()[0]
    event_align.set()

    info = StreamInfo(name, 'EEG', n_channels, srate, 'float32', 'myuid34234')
    outlet = StreamOutlet(info)
    while True:
        holder_1 = np.empty((chunk_size,sample_1.shape[0],))
        holder_2 = np.empty((chunk_size,sample_2.shape[0],))
        for i in range(chunk_size):
            sample_1, timestamp_1 = inlet_ECR.pull_sample()
            sample_2, timestamp_2 = inlet_FCR.pull_sample()
            sample_1 = np.array(sample_1).reshape(-1)
            sample_2 = np.array(sample_2).reshape(-1)
            holder_1[i,:] = sample_1
            holder_2[i,:] = sample_2

        array_data = np.concatenate((holder_1[:,0:64],holder_2[:,0:64]),axis = 1)

        dead_chan = np.where(np.nan_to_num(np.mean(array_data,axis = 0),nan=0,posinf=0,neginf=0) ==0)[0]
        for idx in dead_chan:
            array_data[:,idx] = np.zeros(array_data.shape[0])

        array_data_filt = filt_GRID(array_data,20,500,2000,)
        array_data_scaled = np.nan_to_num(array_data_filt/MVC_vals ,nan=0,posinf=0,neginf=0)

        graph_instance = ML_training.vanilla_training.helper.generate_grid_graph_2D_infer(array_data_scaled, crds_scaled, edge_idx, edge_feat, dead_chan = dead_chan)
        batch_items = 1
        inputs = graph_instance.to(device)
        num_nodes = [graph_instance.num_nodes]
        preds,_ = net(inputs,batch_items, num_nodes)
        decoded_gesture = preds.argmax(dim=1).cpu().numpy().flatten()[0]
        # decoded_gesture = int(np.random.choice([0,1,2,3,4,5,6,7,8,9]))
        outlet.push_sample([decoded_gesture])
        if event_term.is_set():
            print("Stopping decoder")
            break
     
def background_inference_evaluation(model_pth, T0, event_start, event_term,event_align, MVC_arr, inp_dim, dump_path):
    name = "decoder"
    n_channels = 1
    srate = 20
    chunk_size = inp_dim

    SAMPLING_TOL = 1e-5
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/ nyq, highcut/ nyq], btype='band', analog=False)
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y


    def filt_GRID(data, lowcut, highcut, fs, order=3, notch_fs = 50, notch_q = 30):
        filt_out = np.zeros_like(data)
        for i in range(data.shape[1]):
            filt_out[:,i] = notch_filter(butter_bandpass_filter(data[:,i], lowcut, highcut, fs, order=order), notch_fs, fs, notch_q)
        return filt_out

    def notch(notch_freq, samp_freq, quality_factor=30):
        b, a = iirnotch(notch_freq, quality_factor, samp_freq)
        return b, a

    def notch_filter(data, notch_fs, fs, q=30):
        b, a = notch(notch_fs, fs, q)
        y = filtfilt(b, a, data)
        return y

    ECR_MVC = MVC_arr["ECR"]
    FCR_MVC = MVC_arr["FCR"]
    MVC_vals = np.concatenate((ECR_MVC,FCR_MVC)).reshape(1,-1)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    CHKPT = torch.load(model_pth) 
    
    if CHKPT['args'].model == 'GCN':
        net = EMG_GCN(CHKPT['args'])
    elif CHKPT['args'].model == 'GAT':
        net = EMG_GAT(CHKPT['args'])
    elif CHKPT['args'].model == 'CNN':
        net = EMG_CNN(CHKPT['args'])
    else:
        print('model not found')
        raise NotImplementedError
    
    net.load_state_dict(CHKPT['model'])
    net.to(device)
    net.eval()
    
    print("finding stream")
    stream_ECR = resolve_stream('name', 'FCR')
    for info in stream_ECR:
        print('name: ', info.name())
        print('channel count:', info.channel_count())
        print('sampling rate:', info.nominal_srate())
        print('type: ', info.type())
    stream_FCR = resolve_stream('name', 'ECR')
    for info in stream_FCR:
        print('name: ', info.name())
        print('channel count:', info.channel_count())
        print('sampling rate:', info.nominal_srate())
        print('type: ', info.type())

    inlet_ECR = StreamInlet(stream_ECR[0])
    inlet_FCR = StreamInlet(stream_FCR[0])    
    sample_1, timestamp_1 = inlet_ECR.pull_sample()
    sample_2, timestamp_2 = inlet_FCR.pull_sample()
    timestamp_1 += inlet_ECR.time_correction()
    timestamp_2 += inlet_FCR.time_correction()
    while abs(timestamp_1 - timestamp_2) > SAMPLING_TOL:
        if timestamp_1 - timestamp_2 > SAMPLING_TOL:
            while timestamp_1 - timestamp_2 > SAMPLING_TOL:
                sample_2, timestamp_2 = inlet_FCR.pull_sample()
        elif timestamp_2 - timestamp_1  > SAMPLING_TOL:
            while timestamp_2 - timestamp_1 > SAMPLING_TOL:
                sample_1, timestamp_1 = inlet_ECR.pull_sample()
    sample_1 = np.array(sample_1).reshape(-1)
    sample_2 = np.array(sample_2).reshape(-1)
    event_start.set()
    print("sending signal for alignment")
    """
    Sampling loop to fix trigs, 10000 samples
    """
    samples_to_fix = 10000
    trig_1 = np.empty((samples_to_fix))
    trig_2 = np.empty((samples_to_fix))
    for i in range(samples_to_fix):
        sample_1, timestamp_1 = inlet_ECR.pull_sample()
        sample_2, timestamp_2 = inlet_FCR.pull_sample()
        trig_1[i] = sample_1[-3]#np.array(sample_1).reshape(-1)
        trig_2[i] = sample_2[-3]#np.array(sample_2).reshape(-1)
    # plt.plot(trig_1)
    # plt.plot(trig_2)
    # plt.savefig('align_in.png')
    # plt.close()
    flag, gap = align_trig(trig_1, trig_2)
    if flag == 0:
        for i in range(gap):
            sample_2, timestamp_2 = inlet_FCR.pull_sample()
    if flag == 1:
        for i in range(gap):
            sample_1, timestamp_1 = inlet_ECR.pull_sample()

    trig_1 = np.empty((samples_to_fix))
    trig_2 = np.empty((samples_to_fix))
    for i in range(samples_to_fix):
        sample_1, timestamp_1 = inlet_ECR.pull_sample()
        sample_2, timestamp_2 = inlet_FCR.pull_sample()
        trig_1[i] = sample_1[-3]#np.array(sample_1).reshape(-1)
        trig_2[i] = sample_2[-3]#np.array(sample_2).reshape(-1)
    # plt.plot(trig_1)
    # plt.plot(trig_2)
    # plt.savefig('align_out.png')
    # plt.close()

    sample_1 = np.array(sample_1).reshape(-1)
    sample_2 = np.array(sample_2).reshape(-1)
    sample_3 = np.array([0]).reshape(-1)

    print("stream Time", time.time()-T0)
    holder_1 = np.empty((chunk_size,sample_1.shape[0],))
    holder_2 = np.empty((chunk_size,sample_2.shape[0],))
    holder_3 = np.empty((chunk_size,sample_3.shape[0],))
    for i in range(chunk_size):
        sample_1, timestamp_1 = inlet_ECR.pull_sample()
        sample_2, timestamp_2 = inlet_FCR.pull_sample()
        sample_1 = np.array(sample_1).reshape(-1)
        sample_2 = np.array(sample_2).reshape(-1)
        sample_3 = np.array([0]).reshape(-1)
        holder_1[i,:] = sample_1
        holder_2[i,:] = sample_2

    save_path = os.path.join(dump_path,'eval')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f_create_time = str(int(time.time()))
    with h5py.File(save_path+'\\'+f_create_time+'.h5',"a") as EMG_file:
        EMG_file.create_dataset("ECR", data = holder_1,compression = 'gzip',
                                maxshape=(None,sample_1.shape[0],), chunks = (chunk_size,sample_1.shape[0],))
        EMG_file.create_dataset("FCR", data = holder_2,compression = 'gzip',
                                maxshape=(None,sample_2.shape[0]), chunks = (chunk_size,sample_2.shape[0],))
        EMG_file.create_dataset("pred", data = holder_3,compression = 'gzip',
                                maxshape=(None,sample_3.shape[0]), chunks = (chunk_size,sample_3.shape[0],))
    EMG_file.close()
    
    event_align.set()
    
    crds = ML_training.vanilla_training.helper.pkl_load("./ML_training/graph_gen/crd_info_2grid_ECR_FCR.pkl")
    spacing,neigh_dist = 8.75, 12
    crds_scaled = crds * spacing
    edge_idx, edge_feat = ML_training.vanilla_training.helper.get_neigh_2D(crds_scaled, neigh_dist)
    array_data = np.concatenate((holder_1[:,0:64],holder_2[:,0:64]),axis = 1)

    dead_chan = np.where(np.nan_to_num(np.mean(array_data,axis = 0),nan=0,posinf=0,neginf=0) ==0)[0]
    for idx in dead_chan:
        array_data[:,idx] = np.zeros(array_data.shape[0])
    array_data_filt = filt_GRID(array_data,20,500,2000,)
    array_data_scaled = array_data_filt/MVC_vals 
    graph_instance = ML_training.vanilla_training.helper.generate_grid_graph_2D_infer(array_data_scaled, crds_scaled, edge_idx, edge_feat, dead_chan = dead_chan)
    batch_items = 1
    inputs = graph_instance.to(device)
    num_nodes = [graph_instance.num_nodes]
    preds,_ = net(inputs,batch_items, num_nodes)
    decoded_gesture = preds.argmax(dim=1).cpu().numpy().flatten()[0]
    event_align.set()

    info = StreamInfo(name, 'EEG', n_channels, srate, 'float32', 'myuid34234')
    outlet = StreamOutlet(info)
    while True:
        holder_1 = np.empty((chunk_size,sample_1.shape[0],))
        holder_2 = np.empty((chunk_size,sample_2.shape[0],))
        for i in range(chunk_size):
            sample_1, timestamp_1 = inlet_ECR.pull_sample()
            sample_2, timestamp_2 = inlet_FCR.pull_sample()
            sample_1 = np.array(sample_1).reshape(-1)
            sample_2 = np.array(sample_2).reshape(-1)
            holder_1[i,:] = sample_1
            holder_2[i,:] = sample_2

        array_data = np.concatenate((holder_1[:,0:64],holder_2[:,0:64]),axis = 1)

        dead_chan = np.where(np.nan_to_num(np.mean(array_data,axis = 0),nan=0,posinf=0,neginf=0) ==0)[0]
        for idx in dead_chan:
            array_data[:,idx] = np.zeros(array_data.shape[0])

        array_data_filt = filt_GRID(array_data,20,500,2000,)
        array_data_scaled = np.nan_to_num(array_data_filt/MVC_vals ,nan=0,posinf=0,neginf=0)

        graph_instance = ML_training.vanilla_training.helper.generate_grid_graph_2D_infer(array_data_scaled, crds_scaled, edge_idx, edge_feat, dead_chan = dead_chan)
        batch_items = 1
        inputs = graph_instance.to(device)
        num_nodes = [graph_instance.num_nodes]
        preds,_ = net(inputs,batch_items, num_nodes)
        decoded_gesture = preds.argmax(dim=1).cpu().numpy().flatten()[0]
        decoded_gesture = int(np.random.choice([0,1,2,3]))
        outlet.push_sample([decoded_gesture])
        if event_term.is_set():
            print("Stopping decoder")
            break
     
def align_trig(trig_1, trig_2):
    
    trig_1 = abs(trig_1-1)
    trig_2 = abs(trig_2-1)
    diff_vector = np.array(abs(trig_1-trig_2),dtype= int)
    sig_len = diff_vector.shape[0]
    
    first_occur = np.where(diff_vector == 1)[0][0]
    
    idx = first_occur
    while idx<sig_len:
        if diff_vector[idx] == 0:
            second_occur = idx
            idx = sig_len + 1
        idx+=1
    gap_len = second_occur - first_occur
    if trig_1[first_occur]-trig_2[first_occur] < 0:
        trig_1 = trig_1[gap_len:]
        flag = 0
    elif trig_1[first_occur]-trig_2[first_occur] > 0:
        trig_2 = trig_2[gap_len:]
        flag = 1
    else:
        print("dodgy alignment",first_occur, second_occur, )
        plt.plot(trig_1)
        plt.plot(trig_2)
        plt.savefig('dodgy align.png')
        plt.close()
        flag = 2
        raise NotImplementedError
    return flag, gap_len
