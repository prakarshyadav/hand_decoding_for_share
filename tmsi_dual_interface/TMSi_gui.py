# import sys
# sys.path.insert(0, 'tmsi_dual_interface')

import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import sys, time
from tmsi_dual_interface.init_2_TMSi import assign_devices
from PySide2 import QtWidgets
from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_writer import FileWriter, FileFormat
from tmsi_dual_interface.tmsi_libraries.TMSiPlotters.gui import PlottingGUI
from tmsi_dual_interface.tmsi_libraries.TMSiPlotters.plotters import PlotterFormat

from tmsi_dual_interface.tmsi_libraries.TMSiProcessing import filters

class TMSi_GUI(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.title('Dual-TMSi setup')
        self.geometry('600x600')
        self.UNI_list =  list(range(0,64))
        self.BIP_list = []
        self.AUX_list = []
        self.samp_rate = 2000
        self.trig_flag = 1

        self.tmsi_1_name = tk.StringVar()
        self.tmsi_2_name = tk.StringVar()

        self.stop_signal = False

        self.lbl0 = ttk.Label(self, text='Label for TMSi 1 (Lower serial number)')
        self.lbl0.pack(fill='x', expand=True)
        self.t0 = tk.Entry(self, textvariable=self.tmsi_1_name)
        self.t0.insert(0, "FCR")
        self.t0.pack(fill='x', expand=True)
        self.t0.focus()
        self.lbl0.place(x=10, y=10)
        self.t0.place(x=260, y=10)

        self.lbl1 = ttk.Label(self, text='Label for TMSi 2 (Higher serial number)')
        self.lbl1.pack(fill='x', expand=True)
        self.t1 = tk.Entry(self, textvariable=self.tmsi_2_name)
        self.t1.insert(0, "ECR")
        self.t1.pack(fill='x', expand=True)
        self.t1.focus()
        self.lbl1.place(x=10, y=35)
        self.t1.place(x=260, y=35)

        self.tmsi_activate_button = tk.Button(self, text='Start Devices', bg ='yellow')
        self.tmsi_activate_button['command'] = self.init_tmsi
        self.tmsi_activate_button.pack()
        self.tmsi_activate_button.place(x=400, y=35)
        
        self.grid_check = tk.IntVar()
        self.BIP_check = tk.IntVar()
        self.AUX_check = tk.IntVar()
        self.fs_check = tk.IntVar()
        self.trig_check = tk.IntVar()



        self.channel_warn = ttk.Label(self, text='NOTE: Enter channel number before checking the box || Use "None" to not include channel')
        self.channel_warn.pack(fill='x', expand=True)
        self.channel_warn.place(x=10, y=70)

        self.c1 = tk.Checkbutton(self, text='GRID',variable=self.grid_check, onvalue=1, offvalue=0, command= self.set_grid)
        self.c1.pack()
        self.c1.place(x=300, y=90)

        self.c2 = tk.Checkbutton(self, text='BIP',variable=self.BIP_check, onvalue=1, offvalue=0, command=self.set_bip)
        self.c2.pack()
        self.c2.place(x=300, y=110)

        self.c3 = tk.Checkbutton(self, text='AUX',variable=self.AUX_check, onvalue=1, offvalue=0, command=self.set_aux)
        self.c3.pack()
        self.c3.place(x=300, y=130)

        self.c4 = tk.Checkbutton(self, text='Sample rate',variable=self.fs_check, onvalue=1, offvalue=0, command=self.set_fs)
        self.c4.pack()
        self.c4.place(x=300, y=150)

        self.c5 = tk.Checkbutton(self, text='Triggers',variable=self.trig_check, onvalue=1, offvalue=0, command= self.set_trig)
        self.c5.pack()
        self.c5.place(x=300, y=170)


        self.grid_num = tk.StringVar()
        self.lbl2 = ttk.Label(self, text='Grid Channels')
        self.lbl2.pack(fill='x', expand=True)
        self.t2 = tk.Entry(self, textvariable=self.grid_num)
        self.t2.insert(0, "64")
        self.t2.pack(fill='x', expand=True)
        self.t2.focus()
        self.lbl2.place(x=10, y=93)
        self.t2.place(x=150, y=93)

        self.bip_num = tk.StringVar()
        self.lbl3 = ttk.Label(self, text='BIP Channels')
        self.lbl3.pack(fill='x', expand=True)
        self.t3 = tk.Entry(self, textvariable=self.bip_num)
        self.t3.insert(0, "None")
        self.t3.pack(fill='x', expand=True)
        self.t3.focus()
        self.lbl3.place(x=10, y=113)
        self.t3.place(x=150, y=113)

        self.aux_num = tk.StringVar()
        self.lbl4 = ttk.Label(self, text='AUX Channels')
        self.lbl4.pack(fill='x', expand=True)
        self.t4 = tk.Entry(self, textvariable=self.aux_num)
        self.t4.insert(0, "None")
        self.t4.pack(fill='x', expand=True)
        self.t4.focus()
        self.lbl4.place(x=10, y=133)
        self.t4.place(x=150, y=133)

        self.fs_num = tk.StringVar()
        self.lbl5 = ttk.Label(self, text='Sampling rate')
        self.lbl5.pack(fill='x', expand=True)
        self.t5 = tk.Entry(self, textvariable=self.fs_num)
        self.t5.insert(0, "2000")
        self.t5.pack(fill='x', expand=True)
        self.t5.focus()
        self.lbl5.place(x=10, y=153)
        self.t5.place(x=150, y=153)

        
        self.trig_num = tk.StringVar()
        self.lbl6 = ttk.Label(self, text='Triggers (0: Off, 1: On)')
        self.lbl6.pack(fill='x', expand=True)
        self.t6 = tk.Entry(self, textvariable=self.trig_num)
        self.t6.insert(0, "1")
        self.t6.pack(fill='x', expand=True)
        self.t6.focus()
        self.lbl6.place(x=10, y=173)
        self.t6.place(x=150, y=173)

        self.tmsi_configure_button = tk.Button(self, text='Configure Channels', bg ='yellow')
        self.tmsi_configure_button['command'] = self.config_chan
        self.tmsi_configure_button.pack()
        self.tmsi_configure_button.place(x=10, y=195)


        self.imp_plot_button = tk.Button(self, text='Get Imp Plot', bg ='yellow')
        self.imp_plot_button['command'] = self.imp_plot
        self.imp_plot_button.pack()
        self.imp_plot_button.place(x=10, y=220)

        self.vis_sig_button = tk.Button(self, text='Visualize signals', bg ='yellow')
        self.vis_sig_button['command'] = self.start_plotter
        self.vis_sig_button.pack()
        self.vis_sig_button.place(x=100, y=220)

        self.vis_heatmap_button = tk.Button(self, text='Visualize heatmap', bg ='yellow')
        self.vis_heatmap_button['command'] = self.start_heatmap
        self.vis_heatmap_button.pack()
        self.vis_heatmap_button.place(x=210, y=220)

        self.start_dump_button = tk.Button(self, text='PUSH TMSi', bg = 'yellow')
        self.start_dump_button['command'] = self.push_tmsi
        self.start_dump_button.pack()
        self.start_dump_button.place(x=210, y=400)

        # self.start_dump_button = tk.Button(self, text='START DUMPING', bg = 'yellow')
        # self.start_dump_button['command'] = self.start_dumping
        # self.start_dump_button.pack()
        # self.start_dump_button.place(x=210, y=300)


        # self.start_stream_button = tk.Button(self, text='START STREAM', bg = 'yellow')
        # self.start_stream_button['command'] = self.start_stream
        # self.start_stream_button.pack()
        # self.start_stream_button.place(x=210, y=250)
        



        # self.stop_button = tk.Button(self, text='STOP', bg = 'red')
        # self.stop_button['command'] = self.stop_recording
        # self.stop_button.pack()
        # self.stop_button.place(x=210, y=350)

    def push_tmsi(self):
        self.destroy()

    def init_tmsi(self):
        self.device_dict = assign_devices(label_1 = str(self.tmsi_1_name.get()), label_2 = str(self.tmsi_2_name.get()))
        showinfo(title='Information', message='Initialized TMSi')
        self.tmsi_activate_button.config(bg = 'green')

    def set_grid(self):
        if str(self.grid_num.get()) == "None":
            self.UNI_list = []
            showinfo(title='Channel Removed', message="GRID not included in config")

        elif int(self.grid_check.get()):
            try:
                num_chan = int(self.grid_num.get())
            except ValueError:
                showinfo(title='Invalid entry', message="Use Int (16,32,64)")
            if num_chan > 0:
                self.UNI_list = list(range(0,num_chan))
            showinfo(title='Grid assigned', message="Setting " + str(num_chan) + " electrode channels")
        else:
            showinfo(title='Grid not set', message="Check the box again and set valid values")

    def set_bip(self):
        if str(self.bip_num.get()) == "None":
            self.BIP_list = []
            showinfo(title='Channel Removed', message="BIP not included in config")

        elif int(self.BIP_check.get()):
            try:
                num_chan = int(self.bip_num.get())
            except ValueError:
                showinfo(title='Invalid entry', message="Use Int (1,2,3,4)")
            if num_chan > 0:
                self.BIP_list = list(range(0,num_chan))
            showinfo(title='BIP assigned', message="Setting " + str(num_chan) + " bipolar channels")
        else:
            showinfo(title='BIP not set', message="Check the box again and set valid values")

    def set_aux(self):
        if str(self.aux_num.get()) == "None":
            self.AUX_list = []
            showinfo(title='Channel Removed', message="AUX not included in config")

        elif int(self.AUX_check.get()):
            try:
                num_chan = int(self.aux_num.get())
            except ValueError:
                showinfo(title='Invalid entry', message="Use Int (1,2,3)")
            if num_chan > 0:
                self.AUX_list = list(range(0,num_chan))
            showinfo(title='AUX assigned', message="Setting " + str(num_chan) + " aux channels")
        else:
            showinfo(title='AUX not set', message="Check the box again and set valid values")

    def set_fs(self):
        if int(self.fs_check.get()):
            try:
                samp_rate = int(self.fs_num.get())
            except ValueError:
                showinfo(title='Invalid entry', message="Use Int (1000,2000,4000)")
            if samp_rate > 0:
                self.samp_rate = samp_rate
            showinfo(title='FS assigned', message="Setting " + str(samp_rate) + " FS")
        else:
            showinfo(title='FS not set', message="Check the box again and set valid values")

    
    def set_trig(self):
        if int(self.trig_check.get()):
            try:
                trig_flag = int(self.trig_num.get())
            except ValueError:
                showinfo(title='Invalid entry', message="Use Int (0,1)")
            if trig_flag == 1 :
                self.trig_flag = trig_flag
            showinfo(title='Triggers enabled', message="Triggers enabled")
        else:
            showinfo(title='Triggers not set', message="Check the box again and set valid values")

    def config_chan(self):
        for key in self.device_dict.keys():
            # self.device_dict[key].set_sample_rate(self.samp_rate)
            self.device_dict[key].set_triggers(self.trig_flag)
            self.device_dict[key].set_channels(self.UNI_list, self.BIP_list, self.AUX_list)
            

        showinfo(title='Information', message='Channels configured to both TMSi')
        self.tmsi_configure_button.config(bg = 'green')

    def start_plotter(self):

        keysList = list(self.device_dict.keys())
        plotter_app = QtWidgets.QApplication.instance()
        if not plotter_app:
            plotter_app = QtWidgets.QApplication(sys.argv)

        filter_appl = filters.RealTimeFilter(self.device_dict[keysList[0]].dev)
        filter_appl.generateFilter(Fc_hp=20, Fc_lp=500)

        window1 = PlottingGUI(plotter_format = PlotterFormat.signal_viewer,
                                  figurename = self.device_dict[keysList[0]].dev_name, 
                                  device = self.device_dict[keysList[0]].dev, 
                                  channel_selection = [0, 1, 2],
                                  filter_app = filter_appl)
        window1.show()

        plotter_app.exec_()
        del plotter_app

        
        plotter_app = QtWidgets.QApplication.instance()
        if not plotter_app:
            plotter_app = QtWidgets.QApplication(sys.argv)

        filter_appl = filters.RealTimeFilter(self.device_dict[keysList[1]].dev)
        filter_appl.generateFilter(Fc_hp=20, Fc_lp=500)

        window1 = PlottingGUI(plotter_format = PlotterFormat.signal_viewer,
                                  figurename = self.device_dict[keysList[1]].dev_name, 
                                  device = self.device_dict[keysList[1]].dev, 
                                  channel_selection = [0, 1, 2],
                                  filter_app = filter_appl)
        window1.show()

        plotter_app.exec_()

        # Quit and delete the Plotter application
        QtWidgets.QApplication.quit()
        del plotter_app
        self.vis_sig_button.config(bg = "green")
        


    def start_heatmap(self):
        grid_type = '8-8-L'
        keysList = list(self.device_dict.keys())
        plotter_app = QtWidgets.QApplication.instance()
        if not plotter_app:
            plotter_app = QtWidgets.QApplication(sys.argv)


        window1 = PlottingGUI(plotter_format = PlotterFormat.heatmap,
                                    figurename = self.device_dict[keysList[0]].dev_name, 
                                    device = self.device_dict[keysList[0]].dev,
                                    tail_orientation = 'down', 
                                    signal_lim = 150,
                                    grid_type = grid_type)
        window1.show()

        plotter_app.exec_()
        del plotter_app

        
        plotter_app = QtWidgets.QApplication.instance()
        if not plotter_app:
            plotter_app = QtWidgets.QApplication(sys.argv)


        window1 = PlottingGUI(plotter_format = PlotterFormat.heatmap,
                                    figurename = self.device_dict[keysList[1]].dev_name, 
                                    device = self.device_dict[keysList[1]].dev,
                                    tail_orientation = 'down', 
                                    signal_lim = 150,
                                    grid_type = grid_type)
        window1.show()

        plotter_app.exec_()

        # Quit and delete the Plotter application
        QtWidgets.QApplication.quit()
        del plotter_app
        self.vis_heatmap_button.config(bg = "green")
        
    def start_stream(self):
        self.start_stream_button.config(bg = "green")
        self.stop_button.config(bg = "yellow")
        keysList = list(self.device_dict.keys())
        stream_1 = FileWriter(FileFormat.lsl, self.device_dict[keysList[0]].dev_name)
        stream_1.open(self.device_dict[keysList[0]].dev)

        stream_2 = FileWriter(FileFormat.lsl, self.device_dict[keysList[1]].dev_name)
        stream_2.open(self.device_dict[keysList[1]].dev)

        self.device_dict[keysList[1]].dev.start_measurement()
        self.device_dict[keysList[0]].dev.start_measurement()
        
        
    def start_dumping(self):
        self.start_stream_button.config(bg = "green")
        self.stop_button.config(bg = "yellow")
        """
        Needs updates to deal with path and dumping data
        """
        # keysList = list(self.device_dict.keys())
        # stream_1 = FileWriter(FileFormat.lsl, self.device_dict[keysList[0]].dev_name)
        # stream_1.open(self.device_dict[keysList[0]].dev)

        # stream_2 = FileWriter(FileFormat.lsl, self.device_dict[keysList[1]].dev_name)
        # stream_2.open(self.device_dict[keysList[1]].dev)

        # self.device_dict[keysList[1]].dev.start_measurement()
        # self.device_dict[keysList[0]].dev.start_measurement()

    def imp_plot(self):
        keysList = list(self.device_dict.keys())
        plotter_app = QtWidgets.QApplication.instance()
        if not plotter_app:
            plotter_app = QtWidgets.QApplication(sys.argv)
        window1 = PlottingGUI(plotter_format = PlotterFormat.impedance_viewer,
                             figurename =  self.device_dict[keysList[0]].dev_name, 
                             device = self.device_dict[keysList[0]].dev, 
                             layout = 'grid')
        window1.show()
        plotter_app.exec_()
        del plotter_app
        plotter_app = QtWidgets.QApplication.instance()
        if not plotter_app:
            plotter_app = QtWidgets.QApplication(sys.argv)
        window2 = PlottingGUI(plotter_format = PlotterFormat.impedance_viewer,
                             figurename =  self.device_dict[keysList[1]].dev_name, 
                             device = self.device_dict[keysList[1]].dev, 
                             layout = 'grid')
        window2.show()
        plotter_app.exec_()
        del plotter_app
        self.imp_plot_button.config(bg = "green")
        



    def stop_recording(self):
        self.stop_signal = True
        self.start_stream_button.config(bg = "yellow")
        self.stop_button.config(bg = "red")

        keysList = list(self.device_dict.keys())
        self.device_dict[keysList[0]].term_tmsi()
        self.device_dict[keysList[1]].term_tmsi()
        showinfo(title='Information', message='Streaming has stopped. \nMay need to restart GUI')


def main():
    tk_trial = TMSi_GUI.APP()
    tk_trial.mainloop()
    return None

if __name__ == "__main__":
    main()