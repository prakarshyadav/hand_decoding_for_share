import subprocess, time, sys, os
from datetime import datetime
# from PySide2 import QtWidgets
import sys
sys.path.insert(0, 'tmsi_dual_interface')
import sys
from os.path import join, dirname, realpath
Example_dir = dirname(realpath(__file__)) # directory of this file
modules_dir = join(Example_dir, '..') # directory with all modules
measurements_dir = join(Example_dir, '../measurements') # directory with all measurements
sys.path.append(modules_dir)
import time

# from PySide2 import QtWidgets

from tmsi_libraries.TMSiSDK import tmsi_device
from tmsi_libraries.TMSiPlotters.gui import PlottingGUI
from tmsi_libraries.TMSiPlotters.plotters import PlotterFormat
from tmsi_libraries.TMSiSDK.device import DeviceInterfaceType, DeviceState, ChannelType, ReferenceMethod, ReferenceSwitch
from tmsi_libraries.TMSiFileFormats.file_writer import FileWriter, FileFormat
from tmsi_libraries.TMSiSDK.error import TMSiError, TMSiErrorCode, DeviceErrorLookupTable
from tmsi_libraries.TMSiSDK import get_config

from tmsi_libraries.TMSiPlotters.gui import PlottingGUI
from tmsi_libraries.TMSiPlotters.plotters import PlotterFormat



class tmsi_start(object):
    #caller for tmsi
    def __init__(self, ):
        self.dev_name = ''
        self.dev_serial = 0

    def init_tmsi(self,id):
        try:
            # Initialise the TMSi-SDK first before starting using it
            tmsi_device.initialize()
            
            # Execute a device discovery. This returns a list   of device-objects for every discovered device.
            discoveryList = tmsi_device.discover(tmsi_device.DeviceType.saga, DeviceInterfaceType.docked, 
                                                DeviceInterfaceType.usb)

            if (len(discoveryList) > 0):
                # Get the handle to the first discovered device.
                self.dev = discoveryList[id]
                self.dev.open()
        
                grid_type = '8-8-L'
                # options:'4-8-L', '6-11-L', '6-11-S', '8-8-L', '8-8-S', '6-11-L-1', '6-11-L-2', '6-11-S-1', '6-11-S-2', '8-8-L-1', '8-8-L-2', '8-8-S-1', '8-8-S-2'
                self.dev.config.triggers = 1
                # Load the HD-EMG channel set and configuration
                print("Device loaded")
                if self.dev.config.num_channels<64:
                    cfg = get_config("saga32_config_textile_grid_" + grid_type)
                else:
                    cfg = get_config("saga64_config_textile_grid_" + grid_type)
                self.dev.load_config(cfg)

        except TMSiError as e:
            print("!!! TMSiError !!! : ", e.code)
            if (e.code == TMSiErrorCode.device_error) :
                print("  => device error : ", hex(self.dev.status.error))
                DeviceErrorLookupTable(hex(self.dev.status.error))
        #     # Create the device object to interface with the SAGA-system.
        #     self.dev = tmsi_device.create(tmsi_device.DeviceType.saga, DeviceInterfaceType.docked, DeviceInterfaceType.usb)
        #     # Find and open a connection to the SAGA-system and print its serial number
        #     self.dev.open()
        #     # Load the EEG channel set and configuration
        #     if self.dev.config.num_channels<64:
        #         cfg = get_config("saga_config_32UNI")
        #     else:
        #         cfg = get_config("saga_config_64UNI")
        #     self.dev.load_config(cfg)
        # except TMSiError as e:
        #     print("!!! TMSiError !!! : ", e.code)
        #     if (e.code == TMSiErrorCode.device_error) :
        #         print("  => device error : ", hex(self.dev.status.error))
        #         DeviceErrorLookupTable(hex(self.dev.status.error))
        return None

    def plot_imp(self, ):
        """To do Clean up"""
        # Check if there is already a plotter application in existence
        plotter_app = QtWidgets.QApplication.instance()
    
        # Initialise the plotter application if there is no other plotter application
        if not plotter_app:
            plotter_app = QtWidgets.QApplication(sys.argv)
        time.sleep(3)
        # Define the GUI object and show it (either a grid layout or head layout may be chosen)
        # Define the GUI object and show it (either a grid layout or head layout may be chosen)
        window = PlottingGUI(plotter_format = PlotterFormat.impedance_viewer,
                             figurename = 'An Impedance Plot', 
                             device = self.dev, 
                             layout = 'grid')
        window.show()
        time.sleep(3)
    
        # Enter the event loop
        plotter_app.exec_()
    
        time.sleep(3)
        # Delete the Impedace plotter application
        del plotter_app

        return None

    def check_sample_rate(self):
        #Returns sample rate of TMSi
        return self.dev.config.base_sample_rate

    def set_sample_rate(self, rate_Hz = 2000):
        #Sets sample rate of TMSi
        self.dev.config.base_sample_rate = rate_Hz
        self.dev.config.set_sample_rate(ChannelType.AUX, 1)
        self.dev.config.set_sample_rate(ChannelType.BIP, 1)
        return None

    def set_referencing_method(self,):
        # Assigns referencing method
        self.dev.config.reference_method = ReferenceMethod.common,ReferenceSwitch.fixed
        return None

    def set_triggers(self,trig_flag=1):
        self.dev.config.triggers = trig_flag

    def get_channels(self,):
        # prints out channel configuration
        ch_list = self.dev.config.channels
        for idx, ch in enumerate(self.dev.channels):
             print('[{0}] : [{1}] in [{2}]'.format(idx, ch.name, ch.unit_name))
        return None

    def set_channels(self, UNI_list = list(range(0,64)), BIP_list = [], AUX_list = [], triggers = 1):
        #Enables all UNI and no BIP/AUX channels
        ch_list = self.dev.config.channels
        UNI_count = 0
        AUX_count = 0
        BIP_count = 0
        for idx, ch in enumerate(ch_list):
            if (ch.type.value == ChannelType.UNI.value):
                if UNI_count in UNI_list:
                    ch.enabled = True
                else:
                    ch.enabled = False
                UNI_count += 1
            elif (ch.type.value == ChannelType.AUX.value):
                if AUX_count in AUX_list:
                    ch.enabled = True
                else:
                    ch.enabled = False
                AUX_count += 1
            elif (ch.type.value == ChannelType.BIP.value):
                if BIP_count in BIP_list:
                    ch.enabled = True
                else:
                    ch.enabled = False
                BIP_count += 1
            else :
                ch.enabled = False
        self.dev.config.channels = ch_list
        print('\nThe active channels are now :')
        for idx, ch in enumerate(self.dev.channels):
                print('[{0}] : [{1}] in [{2}]'.format(idx, ch.name, ch.unit_name))
        # Update sensor information
        self.dev.update_sensors()
        return None

    def init_file_writer(self,out_dir,out_poly5 ):
        #initializes a file writer
        save_path = out_dir +out_poly5
        self.file_writer = FileWriter(FileFormat.poly5, save_path)
        self.file_writer.open(self.dev)
        return None

    def term_file_writer(self):
        #closes the file writer
        self.file_writer.close()
        return None
    
    def start_emg(self, ):
        "method for actually recording data"
        self.dev.start_measurement() 
        return str(datetime.now())
    
    def stop_emg(self, ):
        "method for actually recording data"
        self.dev.stop_measurement()
        stoptime = str(datetime.now())
        return stoptime

    def term_tmsi(self,):
        # enforces closing tmsi
        # self.dev.close()
        
        if self.dev.status.state == DeviceState.connected:
            self.dev.close()
        return None

def initialize_a_device(id):
    dev = tmsi_start()
    dev.init_tmsi(id)
    dev.dev_serial = int(dev.dev._info.ds_serial_number)
    return dev

def assign_devices(label_1, label_2):
    
    dev_1 = initialize_a_device(0)
    dev_2 = initialize_a_device(1)
    
    if dev_1.dev_serial < dev_2.dev_serial:
        dev_1.dev_name = label_1
        dev_2.dev_name = label_2
        device_dict = {label_1:dev_1,
                       label_2:dev_2,
                        }
    elif dev_1.dev_serial > dev_2.dev_serial:
        dev_1.dev_name = label_2
        dev_2.dev_name = label_1
        device_dict = {label_1:dev_2,
                       label_2:dev_1,
                        }
    else:
        print("Big OOPS! Devices have same serial?")
        raise Exception
    return device_dict

if __name__ == "__main__":
    dev_1 = initialize_a_device(0)
    dev_2 = initialize_a_device(1)