import os
import tkinter as tk
from tkinter import ttk

from gifs import gif_gui
from tmsi_dual_interface import TMSi_gui
from recording import rec_gui
from inference import inf_gui
import PySide2
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class APP(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Main interface for gesture decoding')
        self.geometry('500x200')
        
        self.tmsi_init_button = tk.Button(self, text='Initialize TMSi', bg ='yellow')
        self.tmsi_init_button['command'] = self.open_TMSi_GUI
        self.tmsi_init_button.pack()
        self.tmsi_init_button.place(x=300, y=10)

        self.lbltmsi = ttk.Label(self, text='1. Click to initialize TMSi as "tmsi_dev"') 
        self.lbltmsi.pack(fill='x', expand=True)
        self.lbltmsi.place(x=10, y=15)

        self.prompt_init_button = tk.Button(self, text='Initialize Cues', bg ='yellow')
        self.prompt_init_button['command'] = self.open_cues_GUI
        self.prompt_init_button.pack()
        self.prompt_init_button.place(x=300, y=60)
        
        self.lblprompt = ttk.Label(self, text='2. Click to initialize directory for cues and hand type')
        self.lblprompt.pack(fill='x', expand=True)
        self.lblprompt.place(x=10, y=65)

        self.recording_init_button = tk.Button(self, text='Initialize Recorder', bg ='yellow')
        self.recording_init_button['command'] = self.open_recorder_GUI
        self.recording_init_button.pack()
        self.recording_init_button.place(x=300, y=110)

        self.lblrec = ttk.Label(self, text='3. Click to initialize data recording interface (offline)')
        self.lblrec.pack(fill='x', expand=True)
        self.lblrec.place(x=10, y=115)

        self.inference_init_button = tk.Button(self, text='Initialize Inference', bg ='yellow')
        self.inference_init_button['command'] = self.open_inference_GUI
        self.inference_init_button.pack()
        self.inference_init_button.place(x=300, y=160)

        self.lblinf = ttk.Label(self, text='4. Click to initialize ML based decoder (online)')
        self.lblinf.pack(fill='x', expand=True)
        self.lblinf.place(x=10, y=165)

        # self.cue_path = 'gifs\\gray\\female\\young\\left\\fpv\\'
        # self.tmsi_dev = {}
        # window = rec_gui.Recording_GUI(self, self.cue_path, self.tmsi_dev)
        # self.inference_init_button.config(bg = 'green')

    def open_TMSi_GUI(self):
        window = TMSi_gui.TMSi_GUI(self)
        window.grab_set()
        self.wait_window(window)
        self.tmsi_dev = window.device_dict
        self.tmsi_init_button.config(bg = 'green')

    def open_cues_GUI(self):
        window = gif_gui.Cues_GUI(self)
        window.grab_set()
        self.wait_window(window)
        self.cue_path = window.cue_path
        self.prompt_init_button.config(bg = 'green')

    def open_recorder_GUI(self):
        # self.cue_path = 'gifs\\gray\\female\\young\\right\\fpv\\'
        # self.tmsi_dev = {}
        window = rec_gui.Recording_GUI(self, self.cue_path, self.tmsi_dev)
        self.recording_init_button.config(bg = 'green')

    def open_inference_GUI(self):
        # self.cue_path = 'gifs\\gray\\female\\young\\left\\fpv\\'
        # self.tmsi_dev = {}
        window = inf_gui.Inference_GUI(self, self.cue_path, self.tmsi_dev)
        self.inference_init_button.config(bg = 'green')



def main():
    tk_trial = APP()
    tk_trial.mainloop()
    return None

if __name__ == "__main__":
    main()