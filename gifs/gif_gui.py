import os
import tkinter as tk
from tkinter import ttk

class Cues_GUI(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.geometry('600x450')
        self.title('Cue hand selection')

        self.virtualhand_path = tk.StringVar()
        self.lbl10 = ttk.Label(self, text='Path for virtual hand')
        self.lbl10.pack(fill='x', expand=True)
        self.t10 = tk.Entry(self, textvariable=self.virtualhand_path)
        self.t10.insert(0, "gifs\\")
        self.t10.pack(fill='x', expand=True)
        self.t10.focus()
        self.lbl10.place(x=10, y=10)
        self.t10.place(x=200, y=10, width = 100)

        excl_list = ['__init__.py', '__pycache__','gesture_dict.py', 'gif_gui.py','sound','gesture_master_list.py']
        hand_types = []
        for hand_type in os.listdir(self.virtualhand_path.get()):
            if hand_type not in excl_list:
                hand_types.append(hand_type)

        self.race_list = hand_types
        self.race_drop_down = tk.StringVar()
        self.race_drop_down.set(self.race_list[2])
        self.lbl_race = ttk.Label(self, text='Select hand type')
        self.lbl_race.pack(fill='x', expand=True)
        self.lbl_race.place(x=10, y=35)
        self.drop_down_race = tk.OptionMenu(self, self.race_drop_down, *self.race_list)
        self.drop_down_race.pack(fill='x', expand=True)
        self.drop_down_race.place(x= 200, y = 35)

        gender_types = os.path.join(self.virtualhand_path.get(),self.race_drop_down.get())
        self.gender_list = os.listdir(gender_types)
        self.gender_drop_down = tk.StringVar()
        self.gender_drop_down.set(self.gender_list[0])
        self.lbl_gender = ttk.Label(self, text='Select gender')
        self.lbl_gender.pack(fill='x', expand=True)
        self.lbl_gender.place(x=10, y=75)
        self.drop_down_gender = tk.OptionMenu(self, self.gender_drop_down, *self.gender_list)
        self.drop_down_gender.pack(fill='x', expand=True)
        self.drop_down_gender.place(x= 200, y = 75)

        age_types = os.path.join(gender_types,self.gender_drop_down.get())
        self.age_list = os.listdir(age_types)
        self.age_drop_down = tk.StringVar()
        self.age_drop_down.set(self.age_list[0])
        self.lbl_age = ttk.Label(self, text='Select age')
        self.lbl_age.pack(fill='x', expand=True)
        self.lbl_age.place(x=10, y=115)
        self.drop_down_age = tk.OptionMenu(self, self.age_drop_down, *self.age_list)
        self.drop_down_age.pack(fill='x', expand=True)
        self.drop_down_age.place(x= 200, y = 115)

        split_types = os.path.join(age_types,self.age_drop_down.get())
        self.split_list = os.listdir(split_types)
        self.split_drop_down = tk.StringVar()
        self.split_drop_down.set(self.split_list[1])
        self.lbl_split = ttk.Label(self, text='Select hand')
        self.lbl_split.pack(fill='x', expand=True)
        self.lbl_split.place(x=10, y=155)
        self.drop_down_split = tk.OptionMenu(self, self.split_drop_down, *self.split_list)
        self.drop_down_split.pack(fill='x', expand=True)
        self.drop_down_split.place(x= 200, y = 155)

        self.view_types = os.path.join(split_types,self.split_drop_down.get())
        self.view_list = os.listdir(self.view_types)
        self.view_drop_down = tk.StringVar()
        self.view_drop_down.set(self.view_list[0])
        self.lbl_view = ttk.Label(self, text='Select view')
        self.lbl_view.pack(fill='x', expand=True)
        self.lbl_view.place(x=10, y=195)
        self.drop_down_view = tk.OptionMenu(self, self.view_drop_down, *self.view_list)
        self.drop_down_view.pack(fill='x', expand=True)
        self.drop_down_view.place(x= 200, y = 195)

        # orientation_types = os.path.join(self.view_types,self.view_drop_down.get())
        # self.orientation_list = os.listdir(orientation_types)
        # self.orientation_drop_down = tk.StringVar()
        # self.orientation_drop_down.set(self.orientation_list[0])
        # self.lbl_orientation = ttk.Label(self, text='Select orientation')
        # self.lbl_orientation.pack(fill='x', expand=True)
        # self.lbl_orientation.place(x=10, y=235)
        # self.drop_down_orientation = tk.OptionMenu(self, self.orientation_drop_down, *self.orientation_list)
        # self.drop_down_orientation.pack(fill='x', expand=True)
        # self.drop_down_orientation.place(x= 200, y = 235)

        self.update_path_button = tk.Button(self, text='Update path', bg ='yellow')
        self.update_path_button['command'] = self.update_path
        self.update_path_button.pack()
        self.update_path_button.place(x=10, y=300)


        self.push_path_button = tk.Button(self, text='Push path', bg ='yellow')
        self.push_path_button['command'] = self.push_path
        self.push_path_button.pack()
        self.push_path_button.place(x=10, y=390)
        self.cues_path = tk.StringVar()
    
    def update_path(self):
        
        self.lbl1 = ttk.Label(self, text='Path for cues')
        self.lbl1.pack(fill='x', expand=True)
        self.t1 = tk.Entry(self, textvariable=self.cues_path)
        self.t1.delete(0, tk.END)
        self.t1.insert(0, os.path.join('gifs',self.race_drop_down.get(),self.gender_drop_down.get(),self.age_drop_down.get(),self.split_drop_down.get(),self.view_drop_down.get())+'\\')
        self.t1.pack(fill='x', expand=True)
        self.t1.focus()
        self.lbl1.place(x=10, y=340)
        self.t1.place(x=200, y=340, width = 300)
        self.update_path_button.config(bg = 'green')

    def push_path(self):
        self.cue_path = self.cues_path.get()
        self.destroy()