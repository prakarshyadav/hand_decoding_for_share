import os
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk, ImageSequence
import random
import cv2
import numpy as np
import nidaqmx
import nidaqmx.system
from nidaqmx.constants import LineGrouping
import pygame.mixer
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import datetime, time
from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_writer import FileWriter, FileFormat

class Selection_window(tk.Toplevel):
    def __init__(self, parent, gestures):
        super().__init__(parent)

        self.geometry('300x300')
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
        self.x = gestures
        for each_item in range(len(self.x)):
            self.gesture_list.insert(tk.END, self.x[each_item][:-4])
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

class Recording_GUI(tk.Toplevel):
    def __init__(self, parent, cue_path, tmsi_dev):
        super().__init__(parent)

        # self.geometry('%dx%d%+d+%d'%(1000,700,-2500,0))
        self.title('Recording interface')

        self.cue_path = cue_path
        self.tmsi_dev = tmsi_dev
        self.orientation_list = os.listdir(self.cue_path)
        self.orientation_drop_down = tk.StringVar()
        self.orientation_drop_down.set(self.orientation_list[1])
        self.lbl_orientation = ttk.Label(self, text='Select orientation')
        self.lbl_orientation.pack(fill='x', expand=True)
        self.lbl_orientation.place(x=10, y=15)
        self.drop_down_orientation = tk.OptionMenu(self, self.orientation_drop_down, *self.orientation_list)
        self.drop_down_orientation.pack(fill='x', expand=True)
        self.drop_down_orientation.place(x= 150, y = 10)

        self.update_gestures_button = tk.Button(self, text='Update gestures', bg ='yellow')
        self.update_gestures_button['command'] = self.update_gesture_list
        self.update_gestures_button.pack()
        self.update_gestures_button.place(x=250, y=10)

        gesture_types = os.path.join(self.cue_path,self.orientation_drop_down.get())
        self.gestures_list = os.listdir(gesture_types)
        self.lbl_gestures =  ttk.Label(self, text='Available gestures:')
        self.lbl_gestures.pack(fill='x', expand=True)
        self.lbl_gestures.place(x=400, y=15)
        self.gesture_disp = []
        self.selected_gestures = []
        for ii, g_name in enumerate(self.gestures_list):
            self.selected_gestures.append(g_name[:-4])
            self.gesture_disp.append(ttk.Label(self, text= str(ii+1) +'. '+g_name[:-4]))
            self.gesture_disp[ii].pack(fill='x', expand=True)
            self.gesture_disp[ii].place(x=400, y=ii*17+35)
        
        self.select_gestures_button = tk.Button(self, text='Select gestures', bg ='yellow')
        self.select_gestures_button['command'] = self.select_gestures
        self.select_gestures_button.pack()
        self.select_gestures_button.place(x=10, y=40)
        
        self.num_repetitions = tk.StringVar()
        self.lbl_num_repititions = ttk.Label(self, text='Number of repetitions per gesture')
        self.lbl_num_repititions.pack(fill='x', expand=True)
        self.lbl_num_repititions.place(x=10, y=75)
        self.t_num_repititions = tk.Entry(self, textvariable=self.num_repetitions)
        self.t_num_repititions.insert(0, "5")
        self.t_num_repititions.pack(fill='x', expand=True)
        self.t_num_repititions.focus()
        self.t_num_repititions.place(x=200, y=75, width = 100)

        self.hold_duration = tk.StringVar()
        self.lbl_hold = ttk.Label(self, text='Hold time for each repetition (sec)')
        self.lbl_hold.pack(fill='x', expand=True)
        self.lbl_hold.place(x=10, y=100 )
        self.t_hold = tk.Entry(self, textvariable=self.hold_duration)
        self.t_hold.insert(0, "3")
        self.t_hold.pack(fill='x', expand=True)
        self.t_hold.focus()
        self.t_hold.place(x=200, y=100, width = 100)

        self.num_blocks = tk.StringVar()
        self.lbl_blocks = ttk.Label(self, text='Number of blocks')
        self.lbl_blocks.pack(fill='x', expand=True)
        self.lbl_blocks.place(x=10, y=125 )
        self.t_blocks = tk.Entry(self, textvariable=self.num_blocks)
        self.t_blocks.insert(0, "4")
        self.t_blocks.pack(fill='x', expand=True)
        self.t_blocks.focus()
        self.t_blocks.place(x=200, y=125, width = 100)
        
        self.intertrial_rest = tk.StringVar()
        self.lbl_intertrial = ttk.Label(self, text='Inter trial rest (sec)')
        self.lbl_intertrial.pack(fill='x', expand=True)
        self.lbl_intertrial.place(x=10, y=150 )
        self.t_intertrial = tk.Entry(self, textvariable=self.intertrial_rest)
        self.t_intertrial.insert(0, "1")
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
        self.t_cue.insert(0, "1.5")
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

        self.start_training_button = tk.Button(self, text='START TRAINING', bg ='yellow')
        self.start_training_button['command'] = self.start_training
        self.start_training_button.pack()
        self.start_training_button.place(x=10, y=500)

        self.start_recording_button = tk.Button(self, text='START RECORDING', bg ='yellow')
        self.start_recording_button['command'] = self.start_recording
        self.start_recording_button.pack()
        self.start_recording_button.place(x=210, y=500)

        self.shutdown_tmsi_button = tk.Button(self, text='SHUTDOWN TMSi', bg ='yellow')
        self.shutdown_tmsi_button['command'] = self.shutdown_tmsi
        self.shutdown_tmsi_button.pack()
        self.shutdown_tmsi_button.place(x=200, y=650)

        self.participant_ID = tk.StringVar()
        self.lbl_participant_ID = ttk.Label(self, text='Participant ID:')
        self.lbl_participant_ID.pack(fill='x', expand=True)
        self.lbl_participant_ID.place(x=10, y=550)
        self.t_participant_ID = tk.Entry(self, textvariable=self.participant_ID)
        self.t_participant_ID.insert(0, "P_1")
        self.t_participant_ID.pack(fill='x', expand=True)
        self.t_participant_ID.focus()
        self.t_participant_ID.place(x=200, y=550, width = 100)

        self.exp_date = tk.StringVar()
        self.lbl_exp_date = ttk.Label(self, text='Experiment date:')
        self.lbl_exp_date.pack(fill='x', expand=True)
        self.lbl_exp_date.place(x=10, y=575)
        self.t_exp_date = tk.Entry(self, textvariable=self.exp_date)
        self.t_exp_date.insert(0, str(datetime.date.today()))
        self.t_exp_date.pack(fill='x', expand=True)
        self.t_exp_date.focus()
        self.t_exp_date.place(x=200, y=575, width = 100)
        
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

        self.display_params = {
            "n_rep": int(self.num_repetitions.get()),
            "hold": float(self.hold_duration.get()),
            "n_blk": int(self.num_blocks.get()),
            "inter_trial": float(self.intertrial_rest.get()),
            "pre_trial": float(self.pretrial_rest.get()),
            "start_cue": float(self.cue_duration.get()),
            "travel_time": float(self.gesture_speed.get()),
        }
        self.trial_counter.set(str(self.display_params["n_rep"]))
        self.gesture_counter.set(str(self.display_params["n_blk"]*len(self.selected_gestures)))

    def start_training(self):
        self.display_params = {
            "n_rep": 2,
            "hold": float(self.hold_duration.get()),
            "n_blk": 1,
            "inter_trial": float(self.intertrial_rest.get()),
            "pre_trial": float(self.pretrial_rest.get()),
            "start_cue": float(self.cue_duration.get()),
            "travel_time": float(self.gesture_speed.get()),
        }
        self.trial_counter.set(str(self.display_params["n_rep"]))   
        self.gesture_counter.set(str(self.display_params["n_blk"]*len(self.selected_gestures)))
        disp_gestures = []
        for blk in range(self.display_params["n_blk"]):
            random.shuffle(self.selected_gestures)
            disp_gestures+= self.selected_gestures
        self.gif_play_no_rec(disp_gestures)
        self.start_training_button.config(bg = 'green')

    def gif_play_no_rec(self, selected_gestures):
        params = self.display_params
        path = os.path.join(self.cue_path,self.orientation_drop_down.get())
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

        self.update()
        for ii, gesture in enumerate(selected_gestures):
            frames = get_gif(os.path.join(path,gesture)+'.gif')
            cv2.namedWindow("Gesture", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Gesture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            prompt_image = np.ones((300,300,3))
            textsize = cv2.getTextSize(gesture, font, fontScale/2.2, thickness//2,)[0]
            textX = (prompt_image.shape[1] - textsize[0]) // 2
            textY = (prompt_image.shape[0] + textsize[1]) // 2
            prompt_image = cv2.putText(prompt_image, gesture,  (textX, textY), font, fontScale/2.2, color, thickness//2, cv2.LINE_AA)
            cv2.imshow('Gesture', prompt_image)
            key = cv2.waitKey(int(params['pre_trial']*1000))
            cv2.imshow('Gesture', ready_img)
            key = cv2.waitKey(int(params['start_cue']*0.75*1000))
            cv2.imshow('Gesture', go_image)
            key = cv2.waitKey(int(params['start_cue']*0.25*1000))
            half_len = len(frames)//2
        
            for reps in range(params['n_rep']):
                channel1.play(metronome,0)
                for i in range(half_len):
                    cv2.imshow('Gesture', frames[i])
                    key = cv2.waitKey(int(params['travel_time']*1000//half_len))
                key = cv2.waitKey(int(params['hold']*1000))
                channel1.play(metronome,0)
                for i in range(half_len,len(frames)):
                    cv2.imshow('Gesture', frames[i])
                    key = cv2.waitKey(int(params['travel_time']*1000//half_len))
                key = cv2.waitKey(int(params['inter_trial']*1000))
                self.trial_counter.set(str(self.display_params["n_rep"]-reps-1))
                self.update()
            self.gesture_counter.set(str(self.display_params["n_blk"]*len(self.selected_gestures)-ii-1))
            self.update()

        channel2.play(gong,0)
        pygame.mixer.quit()
        pygame.quit()
        cv2.destroyAllWindows()

    def start_recording(self):
        self.display_params = {
            "n_rep": int(self.num_repetitions.get()),
            "hold": float(self.hold_duration.get()),
            "n_blk": int(self.num_blocks.get()),
            "inter_trial": float(self.intertrial_rest.get()),
            "pre_trial": float(self.pretrial_rest.get()),
            "start_cue": float(self.cue_duration.get()),
            "travel_time": float(self.gesture_speed.get()),
        }
        self.trial_counter.set(str(self.display_params["n_rep"]))   
        self.gesture_counter.set(str(self.display_params["n_blk"]*len(self.selected_gestures)))
        disp_gestures = []
        for blk in range(self.display_params["n_blk"]):
            random.shuffle(self.selected_gestures)
            disp_gestures+= self.selected_gestures
        self.gif_play(disp_gestures)
        self.start_recording_button.config(bg = 'green')

    def gif_play(self, selected_gestures):
        params = self.display_params
        path = os.path.join(self.cue_path,self.orientation_drop_down.get())

        save_path = os.path.join('data',self.participant_ID.get(),self.exp_date.get())

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
        self.update()
        self.task.write(0)
        for ii, gesture in enumerate(selected_gestures):
            self.update()
            frames = get_gif(os.path.join(path,gesture)+'.gif')
            cv2.namedWindow("Gesture", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Gesture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            prompt_image = np.ones((300,300,3))
            textsize = cv2.getTextSize(gesture, font, fontScale/2.2, thickness//2,)[0]
            textX = (prompt_image.shape[1] - textsize[0]) // 2
            textY = (prompt_image.shape[0] + textsize[1]) // 2
            prompt_image = cv2.putText(prompt_image, gesture,  (textX, textY), font, fontScale/2.2, color, thickness//2, cv2.LINE_AA)
            cv2.imshow('Gesture', prompt_image)
            key = cv2.waitKey(int(params['pre_trial']*1000))
            cv2.imshow('Gesture', ready_img)
            key = cv2.waitKey(int(params['start_cue']*0.75*1000))
            cv2.imshow('Gesture', go_image)
            key = cv2.waitKey(int(params['start_cue']*0.25*1000))
            half_len = len(frames)//2
        
            if not os.path.exists(os.path.join(save_path,self.orientation_drop_down.get(),gesture)):
                os.makedirs(os.path.join(save_path,self.orientation_drop_down.get(),gesture))
            self.start_tmsi(os.path.join(save_path,self.orientation_drop_down.get(),gesture))
            self.send_pretrial_signal()

            for reps in range(params['n_rep']):
                self.update()
                channel1.play(metronome,0)
                self.task.write(0)
                for i in range(half_len):
                    cv2.imshow('Gesture', frames[i])
                    key = cv2.waitKey(int(params['travel_time']*1000//half_len))
                key = cv2.waitKey(int(params['hold']*1000))
                channel1.play(metronome,0)
                for i in range(half_len,len(frames)):
                    cv2.imshow('Gesture', frames[i])
                    key = cv2.waitKey(int(params['travel_time']*1000//half_len))
                self.task.write(1)
                key = cv2.waitKey(int(params['inter_trial']*1000))
                self.trial_counter.set(str(self.display_params["n_rep"]-reps-1))
                self.update()
                
            self.task.write(1)
            self.stop_tmsi()
            
            self.gesture_counter.set(str(self.display_params["n_blk"]*len(self.selected_gestures)-ii-1))
            self.update()

            if ii<self.display_params["n_blk"]*len(self.selected_gestures)-1:
                showinfo(title='Session', message=gesture+" done. Click ok for " +selected_gestures[ii+1])
            else:
                showinfo(title='Session', message="All sessions finished!. Restart GUI/Start")
        
        self.task.write(1)

        channel2.play(gong,0)
        pygame.mixer.quit()
        pygame.quit()
        cv2.destroyAllWindows()

    def send_pretrial_signal(self):
        time.sleep(1)
        self.task.write(1)
        time.sleep(0.5)
        self.task.write(0)
        time.sleep(0.1)
        self.task.write(1)
        time.sleep(0.5)

    def start_tmsi(self,save_path):
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
        time.sleep(0.5)

    def stop_tmsi(self):
        keysList = list(self.tmsi_dev.keys())
        self.file_writer1.close()
        self.tmsi_dev[keysList[0]].dev.stop_measurement()
        self.file_writer2.close()
        self.tmsi_dev[keysList[1]].dev.stop_measurement()

    def shutdown_tmsi(self):
        keysList = list(self.tmsi_dev.keys())
        self.tmsi_dev[keysList[0]].dev.close()
        self.tmsi_dev[keysList[1]].dev.close()

    def start_MVC(self):
        self.display_params = {
            "n_rep": int(self.num_MVC.get()),
            "hold": float(self.MVC_duration.get())/3,
            "n_blk": 1,
            "inter_trial": float(self.intertrial_rest.get()),
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
            self.start_tmsi(os.path.join(save_path,gesture))
            self.send_pretrial_signal()
            for reps in range(params['n_rep']):
                cv2.imshow('Gesture', ready_img)
                showinfo("Ready for MVC", "START!!!")
                cv2.imshow('Gesture', go_image)
                key = cv2.waitKey(int(params['start_cue']*0.25*1000))
                channel1.play(metronome,0)
                self.task.write(0)
                key = cv2.waitKey(int(params['travel_time']*1000))
                key = cv2.waitKey(int(params['hold']*1000))
                key = cv2.waitKey(int(params['travel_time']*1000))
                channel1.play(metronome,0)
                self.task.write(1)
                key = cv2.waitKey(int(params['inter_trial']*1000))
            self.stop_tmsi()
        
        self.task.write(1)
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
        self.task = nidaqmx.Task("tmsi-rec")
        self.task.do_channels.add_do_chan( daq_name+'/port1/line0:1',line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task.start()
        self.task.write(1)
        # showinfo(title='DAQ started', message="DAQ Channel configured to 0")
        self.start_daq_button.config(bg = "green")

    def select_gestures(self):
        window = Selection_window(self,self.gestures_list)
        window.grab_set()
        self.wait_window(window)
        self.select_gestures_button.config(bg = 'green')
        self.selected_gestures = window.selected_gestures

    def update_gesture_list(self):
        gesture_types = os.path.join(self.cue_path,self.orientation_drop_down.get())
        self.gestures_list = os.listdir(gesture_types)
        for del_lbl in self.gesture_disp:
            del_lbl.destroy()
        self.gesture_disp = [] 
        if len(self.gestures_list) > 0:
            for ii, g_name in enumerate(self.gestures_list):
                self.gesture_disp.append(ttk.Label(self, text= str(ii+1) +'. '+g_name[:-4]))
                self.gesture_disp[ii].pack(fill='x', expand=True)
                self.gesture_disp[ii].place(x=400, y=ii*17+35)
        else:
            showinfo(title='Error', message='No gestures found for this configuration')
        self.update_gestures_button.config(bg = 'green')