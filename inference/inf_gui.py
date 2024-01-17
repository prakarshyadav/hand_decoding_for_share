import os
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import random
from PIL import Image, ImageSequence
import cv2
import numpy as np
import nidaqmx
import nidaqmx.system
from nidaqmx.constants import LineGrouping
import pygame.mixer
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import datetime, time
from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_writer import FileWriter, FileFormat
import multiprocessing
import torch
from gifs.gesture_master_list import GESTURE_KEYS
from inference import decoder_inference, cue_display, proc_MVC

class TrainingInterface(tk.Toplevel):
    def __init__(self, parent, params, tmsi, gif_path, gestures,chkpt_path, daq, MVC_dict, inp_dim):
        super().__init__(parent)

        y_size = 200
        # self.geometry('%dx%d%+d+%d'%(400,y_size,-2500,0))
        self.title('Training interface')
        self.params = params
        self.tmsi_dev = tmsi
        self.gif_path = gif_path
        self.gestures = gestures
        self.task = daq
        self.chkpt_path = chkpt_path
        self.MVC_dict = MVC_dict
        self.inp_dim = inp_dim
        self.disp_gesture = []
        for tup in self.gestures:
            orient, gesture = tup.split(',')#[0], tup.split(',')[1]
            self.disp_gesture.append(orient+','+gesture)
        self.gesture_drop_down = tk.StringVar()
        self.gesture_drop_down.set(self.disp_gesture[1])
        self.lbl_gesture = ttk.Label(self, text='Select gesture')
        self.lbl_gesture.pack(fill='x', expand=True)
        self.lbl_gesture.place(x=10, y=15)
        self.drop_down_gesture = tk.OptionMenu(self, self.gesture_drop_down, *self.disp_gesture)
        self.drop_down_gesture.pack(fill='x', expand=True)
        self.drop_down_gesture.place(x= 150, y = 10)

        self.start_cue_button = tk.Button(self, text='START CUE', bg ='yellow')
        self.start_cue_button['command'] = self.start_cue
        self.start_cue_button.pack()
        self.start_cue_button.place(x=10, y=50)
        self.stop_cue_button = tk.Button(self, text='STOP CUE', bg ='red')
        self.stop_cue_button['command'] = self.stop_cue
        self.stop_cue_button.pack()
        self.stop_cue_button.place(x=200, y=50)

        self.start_decoder_button = tk.Button(self, text='START DECODING', bg ='yellow')
        self.start_decoder_button['command'] = self.start_decoding
        self.start_decoder_button.pack()
        self.start_decoder_button.place(x=10, y=90)
        self.stop_decoder_button = tk.Button(self, text='STOP DECODING', bg ='red')
        self.stop_decoder_button['command'] = self.stop_decoding
        self.stop_decoder_button.pack()
        self.stop_decoder_button.place(x=200, y=90)

    def stop_decoding(self):
        self.event_term_decoding.set()
        self.disp_decoded.join()
        self.event_term.set()
        self.run_model.join()
        self.start_decoder_button.config(bg = 'yellow')     
        self.stop_decoder_button.config(bg = 'red')    

    def start_decoding(self):
        render_dict = self.get_render_figs()
        T0 = time.time()
        
        self.event_term = multiprocessing.Event()
        self.event_start = multiprocessing.Event()
        self.event_alignment_done = multiprocessing.Event()
        self.run_model = multiprocessing.Process(name='lsl_record', target=decoder_inference.background_inference, args = (self.chkpt_path,T0,self.event_start,self.event_term,self.event_alignment_done,self.MVC_dict, self.inp_dim))
        
        self.run_model.start()
        print("pushed process")
        while not self.event_start.is_set():
            time.sleep(0.01)
        self.alignment_trigs(self.event_alignment_done)
        # showinfo(title='Session', message="Please be ready for decoding")
        self.send_pretrial_signal()
        self.event_term_decoding = multiprocessing.Event()
        self.disp_decoded = multiprocessing.Process(name='disp_decoded', target=cue_display.background_decoded_display, args = (render_dict,self.event_term_decoding))
        self.disp_decoded.start()
        
        self.stop_decoder_button.config(bg = 'green')        
        self.start_decoder_button.config(bg = 'red')

    def get_render_figs(self):
        path_list = []
        dict_lookup = []
        for tup in self.gestures:
            path_list.append(os.path.join(self.gif_path,tup.split(',')[0],tup.split(',')[1]+'.gif',))
            dict_lookup.append(tup.split(',')[0]+'_'+tup.split(',')[1])
        render_dict = {}
        for ii, key_name in enumerate(dict_lookup):
            frames = get_gif(path_list[ii])
            render_dict[GESTURE_KEYS[key_name]] = frames[len(frames)//2]
        frames = get_gif(os.path.join(self.gif_path,'mid\\Wrist Extension.gif',))
        render_dict['rest'] = frames[0]

        return render_dict

    def alignment_trigs(self,event_align):
        while not event_align.is_set():
            self.task.write(1)
            time.sleep(0.3)
            self.task.write(0)
            time.sleep(0.2)

    def send_pretrial_signal(self):
        self.task.write(1)
        time.sleep(0.1)
        self.task.write(0)

    def stop_cue(self):
        self.event_term_cue.set()
        self.disp_cue.join()
        self.start_cue_button.config(bg = 'yellow')     
        self.stop_cue_button.config(bg = 'red')          

    def start_cue(self):
        """
        Gestures should come form model classes
        """
        gesture = os.path.join(self.gesture_drop_down.get().split(',')[0],self.gesture_drop_down.get().split(',')[1])
        path = os.path.join(self.gif_path,gesture)
        coordinates = (10,50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        color = (0,0,0)
        thickness = 2
        ready_img = np.zeros((100,100,3))
        go_image = np.zeros((100,100,3))
        ready_img[:,:,-1] = np.ones((100,100))*255
        go_image[:,:,1] = np.ones((100,100))*255
        ready_img = cv2.putText(ready_img, "READY!", coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        go_image = cv2.putText(go_image, "GO!", coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        cue_frames = get_gif(path+'.gif')
        self.event_term_cue = multiprocessing.Event()
        self.disp_cue = multiprocessing.Process(name='disp_cue', target= cue_display.background_cue_display, args = (cue_frames, self.params,self.event_term_cue))
        self.disp_cue.start()
        self.stop_cue_button.config(bg = 'green')        
        self.start_cue_button.config(bg = 'red')

class Selection_window(tk.Toplevel):
    def __init__(self, parent, gestures):
        super().__init__(parent)

        self.geometry('300x600')
        self.title('Select gestures')
        yscrollbar = tk.Scrollbar(self)
        yscrollbar.pack(side = tk.RIGHT, fill = tk.Y)
        label = tk.Label(self,
                    text = "Select the gestures below :  ",
                    font = ("Times New Roman", 10), 
                    padx = 10, pady = 10)
        label.pack()
        self.gesture_list = tk.Listbox(self, selectmode = "multiple", 
                    yscrollcommand = yscrollbar.set)
        self.x = []
        for orient, gesture in gestures:
            self.x.append(orient+',' +gesture)
        for each_item in range(len(self.x)):
            self.gesture_list.insert(tk.END, self.x[each_item])
            self.gesture_list.itemconfig(each_item, bg = "lime")
        yscrollbar.config(command = self.gesture_list.yview)
        btn = tk.Button(self, text='Push selected gesture', command=self.selected_item)
        btn.pack(side='bottom')
        self.gesture_list.pack(padx = 10, pady = 10,
                expand = tk.YES, fill = "both")
        self.selected_gestures= []
    def selected_item(self):
        for i in self.gesture_list.curselection():
            self.selected_gestures.append(self.gesture_list.get(i))
        self.destroy()

def get_gif(path):
    im = Image.open(path)
    frames = [cv2.cvtColor(np.array(frame.convert('RGB')), cv2.COLOR_RGB2BGR) for frame in ImageSequence.Iterator(im)]    
    im.close()
    return frames

class EvaluationInterface(tk.Toplevel):
    def __init__(self, parent, params, tmsi, gif_path, gestures,chkpt_path, daq, MVC_dict, inp_dim, dump_path):
        super().__init__(parent)

        y_size = 600
        # self.geometry('%dx%d%+d+%d'%(1000,y_size,-2500,0))
        self.title('Evaulation interface')
        self.params = params
        self.tmsi_dev = tmsi
        self.gif_path = gif_path
        self.gestures = gestures
        self.task = daq
        self.chkpt_path = chkpt_path
        self.MVC_dict = MVC_dict
        self.inp_dim = inp_dim
        self.dump_path = dump_path
        
        self.lbl_gestures =  ttk.Label(self, text='Evaluating gestures:')
        self.lbl_gestures.pack(fill='x', expand=True)
        self.lbl_gestures.place(x=400, y=15)
        self.gesture_disp = []
        self.selected_gestures = []
        for ii, g_name in enumerate(self.gestures):
            self.selected_gestures.append(g_name)
            self.gesture_disp.append(ttk.Label(self, text= str(ii+1) +'. '+g_name))
            self.gesture_disp[ii].pack(fill='x', expand=True)
            self.gesture_disp[ii].place(x=400, y=ii*17+35)
        
        self.start_eval_button = tk.Button(self, text='Start Eval', bg ='yellow')
        self.start_eval_button['command'] = self.start_eval
        self.start_eval_button.pack()
        self.start_eval_button.place(x=10, y=10)

        self.stop_eval_button = tk.Button(self, text='Stop Eval', bg ='red')
        self.stop_eval_button['command'] = self.stop_eval
        self.stop_eval_button.pack()
        self.stop_eval_button.place(x=150, y=10)

        self.lbl_rem_gestures = ttk.Label(self, text="Remaining gestures",font=('Helvetica 16 bold'))
        self.lbl_rem_gestures.pack(fill='x', expand=True)
        self.lbl_rem_gestures.place(x=600, y=10)
        self.gesture_counter = tk.StringVar()
        self.gesture_counter.set('0')
        self.lbl_gesture_counter = ttk.Label(self, textvariable=self.gesture_counter,font=('Helvetica 30 bold'))
        self.lbl_gesture_counter.pack(fill='x', expand=True)
        self.lbl_gesture_counter.place(x=600, y=55)

        self.lbl_rem_trl = ttk.Label(self, text="Remaining trials",font=('Helvetica 16 bold'))
        self.lbl_rem_trl.pack(fill='x', expand=True)
        self.lbl_rem_trl.place(x=600, y=100)
        self.trial_counter = tk.StringVar()
        self.trial_counter.set('0')
        self.lbl_trial_counter = ttk.Label(self, textvariable=self.trial_counter,font=('Helvetica 30 bold'))
        self.lbl_trial_counter.pack(fill='x', expand=True)
        self.lbl_trial_counter.place(x=600, y=145)

        self.trial_counter.set(str(self.params["n_rep"]))
        self.gesture_counter.set(len(self.selected_gestures))

    def get_render_figs(self):
        path_list = []
        dict_lookup = []
        for tup in self.gestures:
            path_list.append(os.path.join(self.gif_path,tup.split(',')[0],tup.split(',')[1]+'.gif',))
            dict_lookup.append(tup.split(',')[0]+'_'+tup.split(',')[1])
        render_dict = {}
        for ii, key_name in enumerate(dict_lookup):
            frames = get_gif(path_list[ii])
            render_dict[GESTURE_KEYS[key_name]] = frames[len(frames)//2]
        frames = get_gif(os.path.join(self.gif_path,'mid\\Wrist Extension.gif',))
        render_dict['rest'] = frames[0]
        return render_dict

    def start_eval(self):
        frames = self.get_render_figs()
        T0 = time.time()
        self.event_term = multiprocessing.Event()
        self.event_start = multiprocessing.Event()
        self.event_alignment_done = multiprocessing.Event()
        self.run_model = multiprocessing.Process(name='lsl_record', target=decoder_inference.background_inference, 
                                                 args = (self.chkpt_path,T0,self.event_start,self.event_term,self.event_alignment_done,self.MVC_dict, self.inp_dim))
        self.run_model.start()
        while not self.event_start.is_set():
            time.sleep(0.01)
        self.alignment_trigs(self.event_alignment_done)
        # showinfo(title='Session', message="Please be ready for decoding")
        self.send_pretrial_signal()

        self.event_term_decoding = multiprocessing.Event()
        self.disp_decoded = multiprocessing.Process(name='disp_decoded', target=cue_display.background_decoded_display, args = (frames,self.event_term_decoding))
        self.disp_decoded.start()
        
        self.trial_counter.set(str(self.params["n_rep"]))   
        self.gesture_counter.set(str(len(self.selected_gestures)))
        random.shuffle(self.selected_gestures)
        self.gif_play(self.selected_gestures)

        self.event_term_decoding.set()
        self.disp_decoded.join()
        self.event_term.set()
        self.run_model.join()
        self.event_term_writer.set()
        self.write_decoded.join()
        cv2.destroyAllWindows()
        self.stop_eval_button.config(bg = 'green')        
        self.start_eval_button.config(bg = 'red')

    def stop_eval(self):
        self.event_term_decoding.set()
        self.disp_decoded.join()
        self.event_term.set()
        self.run_model.join()
        self.event_term_writer.set()
        self.write_decoded.join()
        cv2.destroyAllWindows()
        self.start_eval_button.config(bg = 'yellow')     
        self.stop_eval_button.config(bg = 'red')         

    def send_pretrial_signal(self):
        time.sleep(1)
        self.task.write(1)
        time.sleep(0.5)
        self.task.write(0)
        time.sleep(0.1)
        self.task.write(1)
        time.sleep(0.5)

    def alignment_trigs(self,event_align):
        while not event_align.is_set():
            self.task.write(1)
            time.sleep(0.3)
            self.task.write(0)
            time.sleep(0.2)

    def gif_play(self, selected_gestures):
        params = self.params
        path = self.gif_path
        save_path = self.dump_path

        self.event_start_writer = multiprocessing.Event()
        self.event_term_writer = multiprocessing.Event()
        self.write_decoded = multiprocessing.Process(name='decoder_writer', target=cue_display.background_lsl_listener, args = (save_path,self.event_term_writer, self.inp_dim,self.event_start_writer,self.MVC_dict, selected_gestures))
        self.write_decoded.start()

        pygame.init()
        pygame.mixer.init()
        channel1 = pygame.mixer.Channel(0) # argument must be int
        channel2 = pygame.mixer.Channel(1)
        metronome = pygame.mixer.Sound('gifs\\sound\\Metronome.wav')
        gong =  pygame.mixer.Sound('gifs\\sound\\gong.wav')
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2.5
        color = (0,0,0)
        thickness = 3
        ready_img = np.zeros((300,300,3))
        go_image = np.zeros((300,300,3))
        prompt_image = np.ones((300,300,3))
        ready_img[:,:,-1] = np.ones((300,300))*255
        textsize = cv2.getTextSize("READY!", font, fontScale, thickness,)[0]
        textX = (ready_img.shape[1] - textsize[0]) // 2
        textY = (ready_img.shape[0] + textsize[1]) // 2
        ready_img = cv2.putText(ready_img, "READY!",  (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA)
        go_image[:,:,1] = np.ones((300,300))*255
        textsize = cv2.getTextSize("GO!", font, fontScale, thickness,)[0]
        textX = (go_image.shape[1] - textsize[0]) // 2
        textY = (go_image.shape[0] + textsize[1]) // 2
        go_image = cv2.putText(go_image, "GO!",  (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA)

        while not self.event_start_writer.is_set():
            time.sleep(0.01)
        print("task sequence")
        showinfo(title='Ready!', message="Prepare for trials")
        self.update()
        self.task.write(0)
        for ii, gesture in enumerate(selected_gestures):
            self.update()
            frames = get_gif(os.path.join(path,gesture.split(',')[0],gesture.split(',')[1])+'.gif')

            cv2.namedWindow("Cues", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Cues", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            prompt_image = np.ones((300,300,3))
            textsize = cv2.getTextSize(gesture, font, fontScale/2.2, thickness//2,)[0]
            textX = (prompt_image.shape[1] - textsize[0]) // 2
            textY = (prompt_image.shape[0] + textsize[1]) // 2
            prompt_image = cv2.putText(prompt_image, gesture,  (textX, textY), font, fontScale/2.2, color, thickness//2, cv2.LINE_AA)
            cv2.imshow('Cues', prompt_image)
            key = cv2.waitKey(int(params['pre_trial']*1000))
            cv2.imshow('Cues', ready_img)
            key = cv2.waitKey(int(params['start_cue']*0.75*1000))
            cv2.imshow('Cues', go_image)
            key = cv2.waitKey(int(params['start_cue']*0.25*1000))
            half_len = len(frames)//2
            
            self.send_pretrial_signal()
            for reps in range(params['n_rep']):
                self.update()
                channel1.play(metronome,0)
                self.task.write(0)
                for i in range(half_len):
                    cv2.imshow('Cues', frames[i])
                    key = cv2.waitKey(int(params['travel_time']*1000//half_len))
                jitter_time = np.random.choice(params['hold'])
                key = cv2.waitKey(int(jitter_time*1000))
                for i in range(half_len,len(frames)):
                    cv2.imshow('Cues', frames[i])
                    key = cv2.waitKey(int(params['travel_time']*1000//half_len))
                channel1.play(metronome,0)
                self.task.write(1)
                jitter_time = np.random.choice(params['inter_trial'])
                key = cv2.waitKey(int(jitter_time*1000))
                self.trial_counter.set(str(self.params["n_rep"]-reps-1))
                self.update()
                
            self.task.write(1)
            
            self.gesture_counter.set(str(len(self.selected_gestures)-ii-1))
            self.update()
            if ii<len(self.selected_gestures)-1:
                showinfo(title='Session', message=gesture+" done. Click ok for " +selected_gestures[ii+1])
            else:
                showinfo(title='Session', message="All sessions finished!. Restart GUI/Start")
        self.task.write(1)
        cv2.destroyWindow('Cues')
        channel2.play(gong,0)
        pygame.mixer.quit()
        pygame.quit()
        self.event_term_writer.set()
        self.write_decoded.join()

class Inference_GUI(tk.Toplevel):
    def __init__(self, parent, cue_path, tmsi_dev):
        super().__init__(parent,)
        y_size = 1000
        # self.geometry('%dx%d%+d+%d'%(1000,y_size,-2500,0))
        self.title('inference interface')

        self.cue_path = cue_path
        self.tmsi_dev = tmsi_dev

        self.gestures_possible = {}
        self.oreintations  = os.listdir(cue_path)
        for oreintation in self.oreintations:
            self.gestures_possible[oreintation] = {}
            gestures_all = os.listdir(os.path.join(cue_path,oreintation))
            for gesture in gestures_all:
                self.gestures_possible[oreintation][gesture[:-4]] = os.path.join(cue_path,oreintation,gesture)
        
        self.lbl_gestures =  ttk.Label(self, text='Available gestures:')
        self.lbl_gestures.pack(fill='x', expand=True)
        self.lbl_gestures.place(x=400, y=15)
        self.gesture_disp = []
        self.selected_gestures = []
        counter = 0
        x_coord = 400
        for ii, oreint_name in enumerate(self.gestures_possible.keys()):
            for jj, g_name in enumerate(self.gestures_possible[oreint_name].keys()):
                y_coord = counter*17+35
                self.selected_gestures.append((oreint_name, g_name))
                self.gesture_disp.append(ttk.Label(self, text= str(counter+1) +'. '+ oreint_name + ': '+g_name))
                self.gesture_disp[counter].pack(fill='x', expand=True)
                self.gesture_disp[counter].place(x=x_coord, y=y_coord)
                counter +=1
        self.gestures_list = self.selected_gestures.copy()

        model_directory = 'ML_training\\vanilla_training\\model_wt\\'
        self.model_dir = tk.StringVar()
        self.lbl_model_dir = ttk.Label(self, text='Model directory')
        self.lbl_model_dir.pack(fill='x', expand=True)
        self.lbl_model_dir.place(x=10, y=10)
        self.t_model_dir = tk.Entry(self, textvariable=self.model_dir)
        self.t_model_dir.insert(0, model_directory)
        self.t_model_dir.pack(fill='x', expand=True)
        self.t_model_dir.focus()
        self.t_model_dir.place(x=100, y=10, width = 275)
        
        model_name = 'p1_p2_train_p3_test_GCN_94.3359375_6.pth'
        self.model_name = tk.StringVar()
        self.lbl_model_name = ttk.Label(self, text='Model Name')
        self.lbl_model_name.pack(fill='x', expand=True)
        self.lbl_model_name.place(x=10, y=30)
        self.t_model_name = tk.Entry(self, textvariable=self.model_name)
        self.t_model_name.insert(0, model_name)
        self.t_model_name.pack(fill='x', expand=True)
        self.t_model_name.focus()
        self.t_model_name.place(x=100, y=30, width = 275)
        
        self.select_gestures_button = tk.Button(self, text='Select gestures', bg ='yellow')
        self.select_gestures_button['command'] = self.select_gestures
        self.select_gestures_button.pack()
        self.select_gestures_button.place(x=150, y=65)

        self.select_model_button = tk.Button(self, text='Update model', bg ='yellow')
        self.select_model_button['command'] = self.update_model
        self.select_model_button.pack()
        self.select_model_button.place(x=10, y=65)

        self.num_repetitions = tk.StringVar()
        self.lbl_num_repititions = ttk.Label(self, text='Number of repetitions per gesture')
        self.lbl_num_repititions.pack(fill='x', expand=True)
        self.lbl_num_repititions.place(x=10, y=100)
        self.t_num_repititions = tk.Entry(self, textvariable=self.num_repetitions)
        self.t_num_repititions.insert(0, "20")
        self.t_num_repititions.pack(fill='x', expand=True)
        self.t_num_repititions.focus()
        self.t_num_repititions.place(x=200, y=100, width = 100)

        self.hold_duration = tk.StringVar()
        self.lbl_hold = ttk.Label(self, text='Hold time for each repetition (sec)')
        self.lbl_hold.pack(fill='x', expand=True)
        self.lbl_hold.place(x=10, y=125 )
        self.t_hold = tk.Entry(self, textvariable=self.hold_duration)
        self.t_hold.insert(0, "2, 2.25, 2.5, 2.75, 3")
        self.t_hold.pack(fill='x', expand=True)
        self.t_hold.focus()
        self.t_hold.place(x=200, y=125, width = 100)
        
        self.intertrial_rest = tk.StringVar()
        self.lbl_intertrial = ttk.Label(self, text='Inter trial rest (sec)')
        self.lbl_intertrial.pack(fill='x', expand=True)
        self.lbl_intertrial.place(x=10, y=150 )
        self.t_intertrial = tk.Entry(self, textvariable=self.intertrial_rest)
        self.t_intertrial.insert(0, "1, 1.25, 1.5, 1.75, 2")
        self.t_intertrial.pack(fill='x', expand=True)
        self.t_intertrial.focus()
        self.t_intertrial.place(x=200, y=150, width = 100)
        
        self.pretrial_rest = tk.StringVar()
        self.lbl_pretrial = ttk.Label(self, text='Pre trial rest duration (sec)')
        self.lbl_pretrial.pack(fill='x', expand=True)
        self.lbl_pretrial.place(x=10, y=175)
        self.t_pretrial = tk.Entry(self, textvariable=self.pretrial_rest)
        self.t_pretrial.insert(0, "1.5")
        self.t_pretrial.pack(fill='x', expand=True)
        self.t_pretrial.focus()
        self.t_pretrial.place(x=200, y=175, width = 100)

        self.cue_duration = tk.StringVar()
        self.lbl_cue = ttk.Label(self, text='Duration of start cue (sec)')
        self.lbl_cue.pack(fill='x', expand=True)
        self.lbl_cue.place(x=10, y=200)
        self.t_cue = tk.Entry(self, textvariable=self.cue_duration)
        self.t_cue.insert(0, "0.75")
        self.t_cue.pack(fill='x', expand=True)
        self.t_cue.focus()
        self.t_cue.place(x=200, y=200, width = 100)

        self.gesture_speed = tk.StringVar()
        self.lbl_speed = ttk.Label(self, text='Travel time of gesture (sec)')
        self.lbl_speed.pack(fill='x', expand=True)
        self.lbl_speed.place(x=10, y=225)
        self.t_speed = tk.Entry(self, textvariable=self.gesture_speed)
        self.t_speed.insert(0, "0.5")
        self.t_speed.pack(fill='x', expand=True)
        self.t_speed.focus()
        self.t_speed.place(x=200, y=225, width = 100)

        self.daq_name = tk.StringVar()
        self.lbl_daq = ttk.Label(self, text='Name of DAQ')
        self.lbl_daq.pack(fill='x', expand=True)
        self.lbl_daq.place(x=10, y=275)
        self.t_daq = tk.Entry(self, textvariable=self.daq_name)
        self.t_daq.insert(0, "Dev1")
        self.t_daq.pack(fill='x', expand=True)
        self.t_daq.focus()
        self.t_daq.place(x=200, y=275, width = 100)

        self.start_daq_button = tk.Button(self, text='START DAQ', bg ='yellow')
        self.start_daq_button['command'] = self.start_DAQ
        self.start_daq_button.pack()
        self.start_daq_button.place(x=200, y=305)

        self.MVC_type = tk.StringVar()
        self.lbl_MVC_type = ttk.Label(self, text='MVC type (FCR/ECR)')
        self.lbl_MVC_type.pack(fill='x', expand=True)
        self.lbl_MVC_type.place(x=10, y=375)
        self.t_MVC_type = tk.Entry(self, textvariable=self.MVC_type)
        self.t_MVC_type.insert(0, "FCR")
        self.t_MVC_type.pack(fill='x', expand=True)
        self.t_MVC_type.focus()
        self.t_MVC_type.place(x=200, y=375, width = 100)
        
        self.num_MVC = tk.StringVar()
        self.lbl_MVC_num = ttk.Label(self, text='Number of trials for MVC')
        self.lbl_MVC_num.pack(fill='x', expand=True)
        self.lbl_MVC_num.place(x=10, y=400)
        self.t_MVC_num = tk.Entry(self, textvariable=self.num_MVC)
        self.t_MVC_num.insert(0, "1")
        self.t_MVC_num.pack(fill='x', expand=True)
        self.t_MVC_num.focus()
        self.t_MVC_num.place(x=200, y=400, width = 100)

        self.MVC_duration = tk.StringVar()
        self.lbl_MVC_duration = ttk.Label(self, text='Duration of each MVC (sec)')
        self.lbl_MVC_duration.pack(fill='x', expand=True)
        self.lbl_MVC_duration.place(x=10, y=425)
        self.t_MVC_duration = tk.Entry(self, textvariable=self.MVC_duration)
        self.t_MVC_duration.insert(0, "3")
        self.t_MVC_duration.pack(fill='x', expand=True)
        self.t_MVC_duration.focus()
        self.t_MVC_duration.place(x=200, y=425, width = 100)

        self.start_MVC_button = tk.Button(self, text='START MVC', bg ='yellow')
        self.start_MVC_button['command'] = self.start_MVC
        self.start_MVC_button.pack()
        self.start_MVC_button.place(x=200, y=450)

        self.update_MVC_button = tk.Button(self, text='UPDATE MVC', bg ='yellow')
        self.update_MVC_button['command'] = self.update_MVC
        self.update_MVC_button.pack()
        self.update_MVC_button.place(x=200, y=480)

        self.start_streaming_button = tk.Button(self, text='START STREAM', bg ='yellow')
        self.start_streaming_button['command'] = self.start_tmsi_lsl
        self.start_streaming_button.pack()
        self.start_streaming_button.place(x=10, y=525)

        self.stop_streaming_button = tk.Button(self, text='STOP STREAM', bg ='red')
        self.stop_streaming_button['command'] = self.stop_tmsi_lsl
        self.stop_streaming_button.pack()
        self.stop_streaming_button.place(x=210, y=525)

        self.start_training_button = tk.Button(self, text='START TRAINING', bg ='yellow')
        self.start_training_button['command'] = self.start_training
        self.start_training_button.pack()
        self.start_training_button.place(x=10, y=575)

        self.start_recording_button = tk.Button(self, text='START EVALUATION', bg ='yellow')
        self.start_recording_button['command'] = self.start_recording
        self.start_recording_button.pack()
        self.start_recording_button.place(x=210, y=575)

        self.participant_ID = tk.StringVar()
        self.lbl_participant_ID = ttk.Label(self, text='Participant ID:')
        self.lbl_participant_ID.pack(fill='x', expand=True)
        self.lbl_participant_ID.place(x=10, y=625)
        self.t_participant_ID = tk.Entry(self, textvariable=self.participant_ID)
        self.t_participant_ID.insert(0, "P_1_eval")
        self.t_participant_ID.pack(fill='x', expand=True)
        self.t_participant_ID.focus()
        self.t_participant_ID.place(x=200, y=625, width = 100)

        self.exp_date = tk.StringVar()
        self.lbl_exp_date = ttk.Label(self, text='Experiment date:')
        self.lbl_exp_date.pack(fill='x', expand=True)
        self.lbl_exp_date.place(x=10, y=650)
        self.t_exp_date = tk.Entry(self, textvariable=self.exp_date)
        self.t_exp_date.insert(0, str(datetime.date.today()))
        self.t_exp_date.pack(fill='x', expand=True)
        self.t_exp_date.focus()
        self.t_exp_date.place(x=200, y=650, width = 100)
        
        self.lbl_rem_gestures = ttk.Label(self, text="Remaining gestures",font=('Helvetica 16 bold'))
        self.lbl_rem_gestures.pack(fill='x', expand=True)
        self.lbl_rem_gestures.place(x=625, y=10)
        self.gesture_counter = tk.StringVar()
        self.gesture_counter.set('0')
        self.lbl_gesture_counter = ttk.Label(self, textvariable=self.gesture_counter,font=('Helvetica 30 bold'))
        self.lbl_gesture_counter.pack(fill='x', expand=True)
        self.lbl_gesture_counter.place(x=625, y=55)

        self.lbl_rem_trl = ttk.Label(self, text="Remaining trials",font=('Helvetica 16 bold'))
        self.lbl_rem_trl.pack(fill='x', expand=True)
        self.lbl_rem_trl.place(x=600, y=100)
        self.trial_counter = tk.StringVar()
        self.trial_counter.set('0')
        self.lbl_trial_counter = ttk.Label(self, textvariable=self.trial_counter,font=('Helvetica 30 bold'))
        self.lbl_trial_counter.pack(fill='x', expand=True)
        self.lbl_trial_counter.place(x=600, y=145)

        self.display_params = {
            "n_rep": int(self.num_repetitions.get()),
            "hold": np.array(self.hold_duration.get().split(','),dtype = float),
            "inter_trial": np.array(self.intertrial_rest.get().split(','),dtype = float),
            "pre_trial": float(self.pretrial_rest.get()),
            "start_cue": float(self.cue_duration.get()),
            "travel_time": float(self.gesture_speed.get()),
        }
        self.trial_counter.set(str(self.display_params["n_rep"]))
        self.gesture_counter.set(str(len(self.selected_gestures)))
                
        self.start_dq_button = tk.Button(self, text='START debug', bg ='yellow')
        self.start_dq_button['command'] = self.dev_debug
        self.start_dq_button.pack()
        self.start_dq_button.place(x=400, y=450)

        # self.start_DAQ()
        # self.update_model()
        # self.select_gestures()
        # self.update_MVC()
        # self.start_recording()
    
    def dev_debug(self):
        for i in range(10):
            self.task.write(0)
            time.sleep(0.2)
            self.task.write(1)
            time.sleep(0.1)
            self.task.write(0)

    def send_pretrial_signal(self):
        time.sleep(1)
        self.task.write(0)
        time.sleep(0.2)
        self.task.write(1)
        time.sleep(0.1)
        self.task.write(0)

    def start_tmsi_lsl(self):
        print("TMSi: Do LSL stream")
        keysList = list(self.tmsi_dev.keys())
        self.stream_1 = FileWriter(FileFormat.lsl, self.tmsi_dev[keysList[0]].dev_name)
        self.stream_1.open(self.tmsi_dev[keysList[0]].dev)

        self.stream_2 = FileWriter(FileFormat.lsl, self.tmsi_dev[keysList[1]].dev_name)
        self.stream_2.open(self.tmsi_dev[keysList[1]].dev)

        self.tmsi_dev[keysList[1]].dev.start_measurement()
        self.tmsi_dev[keysList[0]].dev.start_measurement()
        self.stop_streaming_button.config(bg = 'green')        
        self.start_streaming_button.config(bg = 'red')
        
    def start_tmsi_poly5(self,save_path):
        start_time = time.time()
        save_path1 = os.path.join(save_path,str(start_time)+'_dev1_'+'.poly5')
        save_path2 = os.path.join(save_path,str(start_time)+'_dev2_'+'.poly5')

        keysList = list(self.tmsi_dev.keys())
        self.file_writer1 = FileWriter(FileFormat.poly5, save_path1)
        self.file_writer1.open(self.tmsi_dev[keysList[0]].dev)
        self.tmsi_dev[keysList[0]].dev.start_measurement()
        self.file_writer2 = FileWriter(FileFormat.poly5, save_path2)
        self.file_writer2.open(self.tmsi_dev[keysList[1]].dev)
        self.tmsi_dev[keysList[1]].dev.start_measurement()

    def stop_tmsi_lsl(self):
        keysList = list(self.tmsi_dev.keys())
        self.tmsi_dev[keysList[0]].dev.stop_measurement()
        self.stream_1.close()
        self.tmsi_dev[keysList[1]].dev.stop_measurement()
        self.stream_2.close()
        self.start_streaming_button.config(bg = 'yellow')      
        self.stop_streaming_button.config(bg = 'red')          

    def stop_tmsi_poly5(self):
        keysList = list(self.tmsi_dev.keys())
        self.file_writer1.close()
        self.tmsi_dev[keysList[0]].dev.stop_measurement()
        self.file_writer2.close()
        self.tmsi_dev[keysList[1]].dev.stop_measurement()

    def shutdown_tmsi(self):
        time.sleep(0.2)
        keysList = list(self.tmsi_dev.keys())
        self.tmsi_dev[keysList[0]].dev.close()
        self.tmsi_dev[keysList[1]].dev.close()

    def start_MVC(self):
        self.display_params = {
            "n_rep": int(self.num_MVC.get()),
            "hold": float(self.MVC_duration.get())/3,
            "inter_trial": 1,
            "pre_trial": float(self.pretrial_rest.get()),
            "start_cue": float(self.cue_duration.get()),
            "travel_time": float(self.MVC_duration.get())/3,
        }

        params = self.display_params
        save_path = os.path.join('data',self.participant_ID.get(),self.exp_date.get(),"MVC")
        
        pygame.init()
        pygame.mixer.init()
        channel1 = pygame.mixer.Channel(0) # argument must be int
        channel2 = pygame.mixer.Channel(1)
        metronome = pygame.mixer.Sound('gifs\\sound\\Metronome.wav')
        gong =  pygame.mixer.Sound('gifs\\sound\\gong.wav')
        coordinates = (10,50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        color = (0,0,0)
        thickness = 2
        ready_img = np.zeros((100,100,3))
        go_image = np.zeros((100,100,3))
        ready_img[:,:,-1] = np.ones((100,100))*255
        go_image[:,:,1] = np.ones((100,100))*255
        ready_img = cv2.putText(ready_img, "READY!", coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        go_image = cv2.putText(go_image, "GO!", coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        
        for ii, gesture in enumerate([self.MVC_type.get()]):
            cv2.namedWindow("Gesture", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Gesture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            if not os.path.exists(os.path.join(save_path,gesture)):
                os.makedirs(os.path.join(save_path,gesture))
            self.start_tmsi_poly5(os.path.join(save_path,gesture))
            self.send_pretrial_signal()
            for reps in range(params['n_rep']):
                cv2.imshow('Gesture', ready_img)
                showinfo("Ready for MVC", "START!!!")
                cv2.imshow('Gesture', go_image)
                key = cv2.waitKey(int(params['start_cue']*0.25*1000))
                channel1.play(metronome,0)
                self.task.write(1)
                key = cv2.waitKey(int(params['travel_time']*1000))
                key = cv2.waitKey(int(params['hold']*1000))
                key = cv2.waitKey(int(params['travel_time']*1000))
                channel1.play(metronome,0)
                self.task.write(0)
                key = cv2.waitKey(int(params['inter_trial']*1000))
            self.stop_tmsi_poly5()
        
        self.task.write(0)
        channel2.play(gong,0)
        pygame.mixer.quit()
        pygame.quit()
        cv2.destroyAllWindows()
        self.start_MVC_button.config(bg = 'green')

    def start_DAQ(self):
        """
        Triggers are read as opposite, 1 is 0 and 0 is 1 in TMSi
        """
        daq_name = self.daq_name.get()
        self.task = nidaqmx.Task("tmsi")
        self.task.do_channels.add_do_chan( daq_name+'/port1/line0:1',line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task.start()
        self.task.write(0)
        # showinfo(title='DAQ started', message="DAQ Channel configured to 0")
        self.start_daq_button.config(bg = "green")

    def select_gestures(self):
        window = Selection_window(self,self.gestures_list)
        window.grab_set()
        self.wait_window(window)
        self.select_gestures_button.config(bg = 'green')
        self.selected_gestures = window.selected_gestures
        
    def update_model(self):
        self.CHKPT = torch.load(self.model_dir.get() +self.model_name.get()) 
        self.gestures_possible  = self.CHKPT['gesture_key']
        self.gesture_disp = []
        self.selected_gestures = []
        counter = 0
        x_coord = 400
        for orient_gesture_pair in self.gestures_possible.keys():
            y_coord = counter*17+35
            oreint_name, g_name = orient_gesture_pair.split('_')
            self.selected_gestures.append((oreint_name, g_name))
            self.gesture_disp.append(ttk.Label(self, text= str(counter+1) +'. '+ oreint_name + ': '+g_name))
            self.gesture_disp[counter].pack(fill='x', expand=True)
            self.gesture_disp[counter].place(x=x_coord, y=y_coord)
            counter +=1
        self.gestures_list = self.selected_gestures.copy()
        self.select_model_button.config(bg = 'green')

    def start_training(self):
        self.display_params = {
            "n_rep": int(self.num_repetitions.get()),
            "hold": np.array(self.hold_duration.get().split(','),dtype = float),
            "inter_trial": np.array(self.intertrial_rest.get().split(','),dtype = float),
            "pre_trial": float(self.pretrial_rest.get()),
            "start_cue": float(self.cue_duration.get()),
            "travel_time": float(self.gesture_speed.get()),
        }

        inp_dim = self.CHKPT['args'].input_feat_dim
        mdl_path = self.model_dir.get() +self.model_name.get()
        window = TrainingInterface(self,self.display_params, self.tmsi_dev,self.cue_path,self.selected_gestures,mdl_path,self.task,self.MVC_dict, inp_dim)
        # window.grab_set()
        # self.wait_window(window)
        self.start_training_button.config(bg = 'green')
        cv2.destroyAllWindows()

    def update_MVC(self):
        self.update_MVC_button.config(bg = 'green')
        path_MVC = os.path.join('data\\',self.participant_ID.get(),self.exp_date.get())
        self.MVC_dict = proc_MVC.proc_MVC_online(path_MVC)

    def start_recording(self):
        self.display_params = {
            "n_rep": int(self.num_repetitions.get()),
            "hold": np.array(self.hold_duration.get().split(','),dtype = float),
            "inter_trial": np.array(self.intertrial_rest.get().split(','),dtype = float),
            "pre_trial": float(self.pretrial_rest.get()),
            "start_cue": float(self.cue_duration.get()),
            "travel_time": float(self.gesture_speed.get()),
        }
        inp_dim = self.CHKPT['args'].input_feat_dim
        mdl_path = self.model_dir.get() +self.model_name.get()
        dump_path = os.path.join('data\\',self.participant_ID.get(),self.exp_date.get())
        window = EvaluationInterface(self,self.display_params, self.tmsi_dev,self.cue_path,self.selected_gestures,mdl_path,self.task,self.MVC_dict, inp_dim, dump_path)
        window.grab_set()
        self.wait_window(window)
        self.start_recording_button.config(bg = 'green')
        