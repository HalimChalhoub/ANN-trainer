#     This file is part of ANN-trainer
#     Copyright (C) 2023  Halim Chalhoub
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
import os
import json
import tkinter as tk
import matplotlib
import numpy as np
import pandas as pd
from tkinter import ttk, filedialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from train_algo_xI1O import deep_train_gd, tansig
from datetime import datetime

def popup(text_string, geo_string):
    top = tk.Toplevel()
    top.geometry(geo_string)
    top.title("")
    top.resizable(False, False)
    top.attributes('-toolwindow', True)
    top.columnconfigure(0, weight=1)
    top_frame = ttk.Frame(top, borderwidth=1)
    top_frame.grab_set()
    top_frame.pack()
    ttk.Label(top_frame, text=text_string).grid(column=0, row=0, pady=5)
    ttk.Button(top_frame, text="Close", command=lambda: top.destroy()).grid(column=0, row=1, pady=5)

class DeepNN:
    def __init__(self, layer_neurons):
        self.layer_neurons = layer_neurons
        self.in_scaling_values = []
        self.out_scaling_values = []
        self.Wn = []
        self.Wb = []
        for i in range(1, len(layer_neurons)):
            wn_matrix = np.ones((layer_neurons[i], layer_neurons[i - 1]))
            self.Wn.append(wn_matrix)
            wb_matrix = np.ones((layer_neurons[i], 1))
            self.Wb.append(wb_matrix)
        self.E_train = []
        self.E_val = []
        self.dW = []
        self.LR = []
        self.NR = []

    def output(self, input_data):
        outputs = []
        for i in range(input_data.shape[0]):
            x = np.array([input_data[i]]).transpose()
            y = x
            for j in range(len(net.Wb)):
                n = self.Wn[j] @ y + self.Wb[j]
                if j < (len(net.Wb) - 1):
                    y = tansig(n)
                else:
                    y = n
            outputs.append(y)
        return np.array(outputs)

    def weight_init_rand(self):
        for i in range(1, len(self.layer_neurons)):
            wn_matrix = np.random.randn(self.layer_neurons[i], self.layer_neurons[i - 1])
            self.Wn[i - 1] = wn_matrix
            wb_matrix = np.random.randn(self.layer_neurons[i], 1)
            self.Wb[i - 1] = wb_matrix

def normalize(data):
    x_min = data.min()
    x_max = data.max()
    r = (data - x_min) * 2 / (x_max - x_min) - 1
    return r

def denormalize(data, min_val, max_val):
    x_min = min_val
    x_max = max_val
    r = data * (x_max - x_min) / 2 + 0.5 * (x_max - x_min) + x_min
    return r

def reading_data():
    cwd = os.getcwd()
    root_tmp = tk.Tk()
    root_tmp.withdraw()
    file_path = filedialog.askopenfilename(title="Select data file", initialdir=cwd)

    global df
    df = pd.read_excel(io=file_path, sheet_name='Data')
    if check_data(df) == 0:
        return

    file.config(state='enabled')
    file.delete(0, "end")
    file.insert(0, file_path)
    file.config(state='disabled')
    update_textbox("Training data file loaded", t=1)
    update_ratio(scale.get())

def check_data(dataframe):
    for col in dataframe.columns.values:
        for i in range(len(col)):
            if i == 0 and not(col[0] == 'x' or col[0] == 'y'):
                popup("data labels do not respect convention", "300x75")
                return 0

    if 1 in [dataframe.dtypes[i] not in ['float64', 'int64'] for i in range(dataframe.columns.__len__())]:
        popup("invalid data entry found", "200x75")
        return 0

    if 1 in [dataframe[col].hasnans for col in dataframe.columns.values]:
        popup("empty data entry found", "200x75")
        return 0

def reading_test_data():
    cwd = os.getcwd()
    root_tmp = tk.Tk()
    root_tmp.withdraw()
    file_path = filedialog.askopenfilename(initialdir=cwd)

    global test_df
    test_df = pd.read_excel(io=file_path, sheet_name='Data')
    if check_data(test_df) == 0:
        return

    testfile.config(state='enabled')
    testfile.delete(0, "end")
    testfile.insert(0, file_path)
    testfile.config(state='disabled')
    update_textbox("Test data file loaded", t=1)

def process_test_data():  # for the testing window
    if not net_trained:
        popup("no network trained/loaded", "200x75")
    elif testfile.get() == '':
        popup("no test file loaded", "150x75")
    else:
        update_textbox(["Testing net of architecture: ", net.layer_neurons], t=1)
        test_plot_inp = test_df.loc[:, test_df.columns.str.startswith('x')]
        test_output_present = test_plot_inp.columns.size < test_df.columns.size

        i_max = test_plot_inp.shape[1]
        if i_max == 1:
            test_plot_inputs = normalize(test_plot_inp.iloc[:, 0])
            test_plot_inputs = np.array([test_plot_inputs]).transpose()
        else:
            test_plot_inputs = []
            for i in range(0, test_plot_inp.shape[1]):
                test_plot_inputs.append(normalize(test_plot_inp.iloc[:, i]))
            test_plot_inputs = np.array(test_plot_inputs).transpose()

        out_raw = net.output(test_plot_inputs).reshape(test_plot_inputs.shape[0])
        if net.layer_neurons[-1] == 1:
            net_output = denormalize(out_raw, net.out_scaling_values[0], net.out_scaling_values[1])
        else:
            net_output = []
            for j in range(0, net.layer_neurons[0]):
                net_output.append(denormalize(out_raw[j], net.out_scaling_values[j][0], net.out_scaling_values[j][1]))

        if test_plot_inp.shape[1] == 1:
            x_data = test_plot_inp.values
            x_label = 'input'
        else:
            x_data = test_plot_inp.index
            x_label = 'input index'

        figure5.clear()
        ax = figure5.add_subplot(111)
        ax.plot(x_data, net_output, linestyle="-", marker=".", color='blue', label='network output')
        num_col = 2
        if test_output_present:
            test_plot_tgt = test_df.loc[:, test_df.columns.str.startswith('y')]
            r = np.corrcoef(net_output, test_plot_tgt.values[:, 0])
            figure5.suptitle('Correlation: R={}'.format(round(r[0, 1], 4)), fontsize='medium',
                             ha='left', x=0.13, y=0.92)
            ax.plot(x_data, test_plot_tgt.values, linestyle="--", color='black', linewidth=0.5, label='test target')
            num_col = 3
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=num_col, borderaxespad=0.3)
        ax.grid('on', which='both', linestyle='--')
        ax.set_xlabel(x_label)
        ax.set_ylabel('output')
        window5.draw()

def unload_data(val):
    global df, inp, tgt, val_inp, val_tgt, train_inputs, train_targets, val_inputs, val_targets,\
        in_size, out_size, plot_inputs, plot_tgt, plot_inp, in_scaling_values, out_scaling_values

    data2.config(state='enabled')
    data2.delete(0, "end")
    if val_data_need.get() == 'with_val':
        training_data = df.sample(frac=(100-val)/100)
        validation_index = [indx for indx in list(df.index) if indx not in list(training_data.index)]
        validation_data = df.iloc[validation_index]
        data2.insert(0, validation_data.shape[0])
        val_inp = validation_data.loc[:, validation_data.columns.str.startswith('x')].sort_index()
        val_tgt = validation_data.loc[:, validation_data.columns.str.startswith('y')].sort_index()
        inp = training_data.loc[:, training_data.columns.str.startswith('x')].sort_index()
        tgt = training_data.loc[:, training_data.columns.str.startswith('y')].sort_index()
    else:
        data2.insert(0, '0')
        inp = df.loc[:, df.columns.str.startswith('x')]
        tgt = df.loc[:, df.columns.str.startswith('y')]
    data2.config(state='disabled')
    data1.config(state='enabled')
    data1.delete(0, "end")
    data1.insert(0, inp.shape[0])
    data1.config(state='disabled')

    plot_inp = df.loc[:, df.columns.str.startswith('x')]
    plot_tgt = df.loc[:, df.columns.str.startswith('y')]

    i_max = inp.shape[1]
    if i_max == 1:
        in_size = 1
        in_scaling_values = (plot_inp.iloc[:, 0].min(), plot_inp.iloc[:, 0].max())
        train_inputs = normalize(inp.iloc[:, 0])
        train_inputs = np.array([train_inputs]).transpose()
        if val_data_need.get() == 'with_val':
            val_inputs = normalize(val_inp.iloc[:, 0])
            val_inputs = np.array([val_inputs]).transpose()
        plot_inputs = normalize(plot_inp.iloc[:, 0])
        plot_inputs = np.array([plot_inputs]).transpose()
    else:
        in_size = inp.shape[1]
        in_scaling_values = []
        train_inputs = []
        plot_inputs = []
        for i in range(0, in_size):
            in_scaling_values.append((plot_inp.iloc[:, i].min(), plot_inp.iloc[:, i].max()))
            train_inputs.append(normalize(inp.iloc[:, i]))
            plot_inputs.append(normalize(plot_inp.iloc[:, i]))
        train_inputs = np.array(train_inputs).transpose()
        plot_inputs = np.array(plot_inputs).transpose()
        if val_data_need.get() == 'with_val':
            val_inputs = []
            for i in range(0, in_size):
                val_inputs.append(normalize(val_inp.iloc[:, i]))
            val_inputs = np.array(val_inputs).transpose()

    out_size = 1
    out_scaling_values = (plot_tgt.iloc[:, 0].min(), plot_tgt.iloc[:, 0].max())
    train_targets = normalize(tgt.iloc[:, 0])
    train_targets = np.array([train_targets]).transpose()
    if val_data_need.get() == 'with_val':
        val_targets = normalize(val_tgt.iloc[:, 0])
        val_targets = np.array([val_targets]).transpose()

def check_parameters():
    global epochs, val_break, lr, mm, seq_nbr
    epochs = epochs_entry.get()
    if len(epochs) == 0 or epochs == 0:
        popup("Error, no max epoch input", "200x75")
        return 0
    try:
        epochs = int(epochs)
    except ValueError:
        popup("Error, max epoch is not an integer", "250x75")
        return 0
    if int(epochs) == 0:
        popup("Error, max epoch should be greater than zero", "300x75")
        return 0
    val_break = val_break_entry.get()
    if len(val_break) == 0:
        popup("Error, no validation break input", "250x75")
        return 0
    try:
        val_break = int(val_break)
    except ValueError:
        popup("Error, validation break is not an integer", "300x75")
        return 0
    lr = lr_entry.get()
    if len(lr) == 0:
        popup("Error, no learning rate input", "200x75")
        return 0
    try:
        lr = float(lr)
    except ValueError:
        popup("Error, learning rate is not a number", "300x75")
        return 0
    mm = mm_entry.get()
    if len(mm) == 0:
        popup("Error, no momentum coefficient input", "300x75")
        return 0
    try:
        mm = float(mm)
    except ValueError:
        popup("Error, momentum coefficient is not a number", "300x75")
        return 0
    if train_type.get() == 'batch':
        seq_nbr = batch.get()
        if len(seq_nbr) == 0:
            popup("Error, no training sequence number precised", "300x75")
            return 0
        else:
            try:
                seq_nbr = int(seq_nbr)
            except ValueError:
                popup("Error, training sequence input is not a number", "300x75")
                return 0

def check_archi():
    if data1.get() == '0':
        popup("No training data", "150x75")
        return
    hl_archi = archi.get()
    if len(hl_archi) == 0:
        popup("Error, no hidden layer added", "200x75")
        return
    hl_archi = list(hl_archi)
    if hl_archi[0] == ',' or hl_archi[-1] == ',':
        popup("Error, wrong hidden layer format", "200x75")
        return
    string = hl_archi[0]
    hl_archi_str = []
    i = 1
    while i < len(hl_archi):
        if hl_archi[i] != ',':
            string += hl_archi[i]
            i += 1
        else:
            hl_archi_str.append(string)
            string = hl_archi[i + 1]
            i += 2
    hl_archi_str.append(string)
    hl_archi = hl_archi_str
    try:
        hl_archi = [int(entry) for entry in hl_archi]
    except ValueError:
        popup("Error, non-numeric as input in hidden layer", "300x75")
        return
    if check_parameters() == 0:
        return
    else:
        unload_data(scale.get())
        create_net(hl_archi)

def create_net(hl_archi):
    hl_archi.insert(0, in_size)
    hl_archi.append(out_size)
    architecture = tuple(hl_archi)
    global net
    net = DeepNN(architecture)
    net.in_scaling_values = in_scaling_values
    net.out_scaling_values = out_scaling_values
    update_textbox("======================================================", t=0)
    if val_data_need.get() == 'no_val':
        update_textbox("Training without a separate validation set", t=1)
    else:
        update_textbox("Training with a separate validation set", t=1)
    update_textbox("======================================================", t=0)
    update_textbox(["Network architecture: ", net.layer_neurons], t=0)
    if train_type.get() == 'single':
        train_single()
    else:
        train_batch()

def train_single():
    global net_trained
    if val_data_need.get() == 'with_val':
        inputs = [train_inputs, val_inputs]
        targets = [train_targets, val_targets]
    else:
        inputs = train_inputs
        targets = train_targets
    net.weight_init_rand()
    a = time.time()
    deep_train_gd(net, inputs, targets, val_data_need.get(), epochs, val_break, lr, mm, textbox, 1)
    b = time.time()
    if (b-a) < 1:
        update_textbox(["Time to train network (in ms) = ", str(round((time.time() - a)*1000, 3))], t=0)
    else:
        update_textbox(["Time to train network (in s) = ", str(round(time.time() - a, 3))], t=0)
    net_trained = 1
    plot_data()

def train_batch():
    global net_trained
    if val_data_need.get() == 'with_val':
        inputs = [train_inputs, val_inputs]
        targets = [train_targets, val_targets]
    else:
        inputs = train_inputs
        targets = train_targets
    global net
    a = time.time()
    net.weight_init_rand()
    deep_train_gd(net, inputs, targets, val_data_need.get(), epochs, val_break, lr, mm, textbox, 0)
    err = net.E_val[-1]
    best_net = net
    for i in range(seq_nbr-1):
        net.weight_init_rand()
        deep_train_gd(net, inputs, targets, val_data_need.get(), epochs, val_break, lr, mm, textbox, 0)
        if net.E_val[-1] < err:
            err = net.E_val[-1]
            best_net = net
    net = best_net
    update_textbox(["Time to train batch (in s) = ", str(round(time.time() - a, 3))], t=0)
    update_textbox(["Lowest normalized training error = ", str(round(net.E_train[-1], 5))], t=0)
    if val_data_need.get() == 'with_val':
        update_textbox(["Lowest normalized validation error = ", str(round(net.E_val[-1], 5))], t=0)
    net_trained = 1
    plot_data()

def retrain():
    if net_trained:
        if data1.get() == '0':
            popup("No training data", "150x75")
        else:
            if val_data_need.get() == 'with_val':
                inputs = [train_inputs, val_inputs]
                targets = [train_targets, val_targets]
            else:
                inputs = train_inputs
                targets = train_targets
            if check_parameters() == 0:
                return
            update_textbox("------------------------------------------------------", t=0)
            if val_data_need.get() == 'no_val':
                update_textbox("Retraining without a separate validation set", t=1)
            else:
                update_textbox("Retraining with a separate validation set", t=1)
            update_textbox("------------------------------------------------------", t=0)
            a = time.time()
            deep_train_gd(net, inputs, targets, val_data_need.get(), epochs, val_break, lr, mm, textbox, 1)
            b = time.time()
            if (b - a) < 1:
                update_textbox(["Time to retrain network (in ms) = ", str(round((time.time() - a) * 1000, 3))], t=0)
            else:
                update_textbox(["Time to retrain network (in s) = ", str(round(time.time() - a, 3))], t=0)
            plot_data()
    else:
        popup("no network trained/loaded", "200x75")

def plot_data():
    ep = np.arange(1, len(net.E_train) + 1)
    out_raw = net.output(plot_inputs).reshape(plot_tgt.shape)
    if tgt.shape[1] == 1:
        net_output = denormalize(out_raw, plot_tgt.iloc[:, 0].min(), plot_tgt.iloc[:, 0].max())
    else:
        net_output = []
        for j in range(0, plot_tgt.shape[1]):
            net_output.append(denormalize(out_raw[j], plot_tgt.iloc[:, j].min(), plot_tgt.iloc[:, j].max()))

    line_x = np.array([min(plot_tgt.values), max(plot_tgt.values)])
    line_y = line_x
    if plot_inp.shape[1] == 1:
        x_data = plot_inp.values
        x_label = 'input'
    else:
        x_data = plot_inp.index
        x_label = 'input index'

    x_data_train = x_data[inp.index]
    net_output_train = net_output[tgt.index]

    figure1.clear()
    figure1.suptitle('The training error evolution', fontsize='medium', ha='left', x=0.2, y=0.93)
    ax1 = figure1.add_subplot(111)
    ax1.plot(ep, net.E_train, label='training data')
    if val_data_need.get() == 'with_val':
        ax1.plot(ep, net.E_val, label='validation data')
    ax1.grid('on', which='both', linestyle='--')
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1, borderaxespad=0.3)
    ax1.set_xlabel('training iteration')
    ax1.set_ylabel('mean squared error')
    ax1.set_yscale('log')
    window1.draw()

    figure2.clear()
    figure2.suptitle('Network response', fontsize='medium', ha='left', x=0.2, y=0.93)
    ax2 = figure2.add_subplot(111)
    ax2.plot(x_data, plot_tgt.values, linestyle="-", color='black', linewidth=0.5, label='target')
    ax2.plot(x_data_train, net_output_train, linestyle="", marker=".", label='training output')
    if val_data_need.get() == 'with_val':
        x_data_val = x_data[val_inp.index]
        net_output_val = net_output[val_tgt.index]
        ax2.plot(x_data_val, net_output_val, linestyle="", marker=".", label='validation output')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=1, borderaxespad=0.3)
    ax2.grid('on', linestyle='--')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('output')
    window2.draw()

    if val_data_need.get() == 'with_val':
        r = np.corrcoef(np.concatenate((tgt.values[:, 0], val_tgt.values[:, 0])),
                        np.concatenate((net_output_train[:, 0], net_output_val[:, 0])))
    else:
        r = np.corrcoef(tgt.values[:, 0], net_output_train[:, 0])

    figure3.clear()
    figure3.suptitle('Correlation: R={}'.format(round(r[0, 1], 4)), fontsize='medium', ha='left', x=0.2, y=0.93)
    ax3 = figure3.add_subplot(111)
    ax3.plot(line_x, line_y, linestyle="--", color='black', linewidth=0.5, label='1:1')
    ax3.plot(tgt.values, net_output_train, linestyle="", marker=".", label='training output')
    if val_data_need.get() == 'with_val':
        net_output_val = net_output[val_tgt.index]
        ax3.plot(val_tgt.values, net_output_val, linestyle="", marker=".", label='validation output')
    ax3.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, borderaxespad=0.3)
    ax3.grid('on',linestyle='--')
    ax3.set_xlabel('network target')
    ax3.set_ylabel('network output')
    window3.draw()

    figure4.clear()
    figure4.suptitle('Average weight updates (absolute value)', fontsize='medium', ha='left', x=0.2, y=0.93)
    ax4 = figure4.add_subplot(111)
    for i in range(len(net.dW)):
        ax4.plot(ep, np.abs(net.dW[i]), label="L{}".format(i+1))
    ax4.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1, borderaxespad=0.3)
    ax4.grid('on', which='both', linestyle='--')
    ax4.set_xlabel('training iteration')
    ax4.set_ylabel('|dW|')
    ax4.set_yscale('log')
    window4.draw()

def save_net():
    if not net_trained:
        popup("no network trained/loaded", "200x75")
        return
    Wn_list = [x.tolist() for x in net.Wn]
    Wb_list = [y.tolist() for y in net.Wb]
    scaling_val = net.out_scaling_values
    json_object = json.dumps(["layer neurons", net.layer_neurons, "Wn", Wn_list,
                              "Wb", Wb_list, "SV", scaling_val], indent=4)
    with open("network.json", "w") as outfile:
        outfile.write(json_object)
    update_textbox("Network saved", t=1)

def load_net():
    global net, net_trained
    cwd = os.getcwd()
    file_path = filedialog.askopenfilename(title="Select network", initialdir=cwd, filetypes=[("JSON", "*.json")])
    with open(file_path, 'r') as openfile:
        json_object = json.load(openfile)
        net = DeepNN(layer_neurons=json_object[1])
        load_Wn = [np.array(x) for x in json_object[3]]
        load_Wb = [np.array(x) for x in json_object[5]]
        net.Wn = np.array(load_Wn)
        net.Wb = np.array(load_Wb)
        net.out_scaling_values = json_object[7]
    net_trained = 1
    update_textbox("======================================================", t=0)
    update_textbox("Network loaded", t=1)
    update_textbox("======================================================", t=0)


if __name__ == "__main__":

    # GUI interface buttons, entry boxes, text box display

    root = tk.Tk()
    root.iconbitmap(default='images/favicon.ico')
    root.title('Artificial Neural Network Trainer')
    root.geometry("1700x800")
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    leftframe = ttk.Frame(root, borderwidth=1, relief='sunken', padding=10)
    leftframe.grid(column=0, row=0, sticky="nsew")
    rightframe = ttk.Frame(root, borderwidth=1, relief='sunken', padding=5)
    rightframe.grid(column=1, row=0, sticky="nsew")

    data1_label = ttk.Label(leftframe, text="Number of training datasets:")
    data1_label.grid(column=0, row=2, sticky='w,e', pady=5)
    data1 = ttk.Entry(leftframe, state='disabled')
    data1.grid(column=1, row=2, pady=5)
    data2_label = ttk.Label(leftframe, text="Number of validation datasets:")
    data2_label.grid(column=0, row=3, sticky='w,e', pady=5)
    data2 = ttk.Entry(leftframe, state='disabled')
    data2.grid(column=1, row=3, pady=5)

    val_data_need = tk.StringVar()
    val_data_need.set('with_val')
    df = pd.DataFrame()
    B_train_file = ttk.Button(leftframe, text='Select training data file', command=lambda: reading_data())
    B_train_file.grid(column=0, row=0, sticky='w,e', pady=5)
    file = ttk.Entry(leftframe, state='disabled')
    file.grid(column=1, columnspan=2, row=0, sticky='w,e', pady=5)

    def no_val():
        scale.config(state='disabled')
        update_ratio(scale.get())

    def with_val():
        scale.config(state='enabled')
        update_ratio(scale.get())

    RB_alldata = ttk.Radiobutton(leftframe, text='Use all data for training', variable=val_data_need, value='no_val',
                                 command=lambda: no_val())
    RB_alldata.grid(column=0, row=1, sticky='w,e', pady=5)
    val_ratio = ttk.Radiobutton(leftframe, text='Create validation\ndata with ratio: ', variable=val_data_need,
                                value='with_val', command=lambda: with_val())
    val_ratio.grid(column=1, row=1, sticky='w,e', pady=5)

    def update_ratio(val):
        val = str(round(float(val)))
        val_ratio['text'] = "Create validation\ndata with ratio: " + val + "%"
        try:
            unload_data(int(val))
        except IndexError:
            return

    scale = ttk.Scale(leftframe, orient='horizontal', length=120, from_=10.0, to=90.0, command=update_ratio)
    scale.grid(column=2, columnspan=1, row=1, sticky='w,e', pady=5)
    scale.set(20)

    ttk.Label(leftframe, text="HL architecture\nseparated by commas: #,#,...").grid(column=0, row=4,
                                                                                    sticky='w,e', pady=5)
    archi = ttk.Entry(leftframe, state='enabled')
    archi.grid(column=1, row=4, pady=5)

    ttk.Label(leftframe, text="Maximum epochs:").grid(column=0, row=5, sticky='w,e', pady=5)
    epochs_entry = ttk.Entry(leftframe, state='enabled')
    epochs_entry.grid(column=1, row=5, pady=5)
    epochs_entry.insert(0, string="100")
    ttk.Label(leftframe, text="Validation break:").grid(column=0, row=6, sticky='w,e', pady=5)
    val_break_entry = ttk.Entry(leftframe, state='enabled')
    val_break_entry.grid(column=1, row=6, pady=5)
    val_break_entry.insert(0, string="6")
    ttk.Label(leftframe, text="Learning rate:").grid(column=0, row=7, sticky='w,e', pady=5)
    lr_entry = ttk.Entry(leftframe, state='enabled')
    lr_entry.grid(column=1, row=7, pady=5)
    lr_entry.insert(0, string="0.2")
    ttk.Label(leftframe, text="Momentum coefficient:").grid(column=0, row=8, sticky='w,e', pady=5)
    mm_entry = ttk.Entry(leftframe, state='enabled')
    mm_entry.grid(column=1, row=8, pady=5)
    mm_entry.insert(0, string="0.1")

    train_type = tk.StringVar()
    net_trained = 0
    train_type.set('single')
    batch = ttk.Entry(leftframe, state='disabled')
    batch.grid(column=2, row=9, sticky='w,e', pady=5)
    RB_single = ttk.Radiobutton(leftframe, text='Single training seq', variable=train_type, value='single',
                                command=lambda: batch.config(state='disabled'))
    RB_single.grid(column=0, row=9, pady=5)
    RB_batch = ttk.Radiobutton(leftframe, text='Multiple training seq: ', variable=train_type, value='batch',
                               command=lambda: batch.config(state='enabled'))
    RB_batch.grid(column=1, row=9, pady=5)

    B_train = ttk.Button(leftframe, text='Train new network', command=lambda: check_archi())
    B_train.grid(column=0, columnspan=3, row=10, sticky='w,e', pady=5)
    B_retrain = ttk.Button(leftframe, text='Retrain current network', command=lambda: retrain())
    B_retrain.grid(column=0, columnspan=3, row=11, sticky='w,e', pady=5)
    B_save = ttk.Button(leftframe, text='Save network', command=lambda: save_net())
    B_save.grid(column=0, columnspan=3, row=12, sticky='w,e', pady=5)
    B_load = ttk.Button(leftframe, text='Load network', command=lambda: load_net())
    B_load.grid(column=0, columnspan=3, row=13, sticky='w,e', pady=5)

    textbox = scrolledtext.ScrolledText(leftframe, width=55, height=13, state='disabled', wrap="word",
                                        background='white', pady=5)
    textbox.grid(row=14, column=0, columnspan=3, sticky='n,e,s,w', pady=5)

    def update_textbox(text, t):
        textbox.config(state='normal')
        if t == 1:
            textbox.insert(tk.INSERT, datetime.now().time().isoformat(timespec='seconds'))
            textbox.insert(tk.INSERT, " - ")
        for i in range(0, text.__len__()):
            textbox.insert(tk.INSERT, text[i])
        textbox.insert(tk.INSERT, "\n")
        textbox.see(tk.END)
        textbox.config(state='disabled')

    B1 = ttk.Button(leftframe, text='Training window', command=lambda: training_win(), state='disabled')
    B1.grid(column=0, columnspan=3, row=15, sticky='w,e', pady=5)
    B2 = ttk.Button(leftframe, text='Testing window', command=lambda: testing_win())
    B2.grid(column=0, columnspan=3, row=16, sticky='w,e', pady=5)

    B_test_file = ttk.Button(rightframe, text='Select testing data file', command=lambda: reading_test_data())
    testfile = ttk.Entry(rightframe, state='disabled')
    B_net_response = ttk.Button(rightframe, text='Network response', command=lambda: process_test_data())

    ttl_train = ttk.Label(rightframe, text="Network training", font="small", borderwidth=1, relief="groove", padding=8)
    ttl_train.grid(column=0, columnspan=2, row=0, pady=5)
    ttl_test = ttk.Label(rightframe, text="Network testing", font="small", borderwidth=1, relief="groove", padding=8)

    # switch between training and testing windows

    def training_win():
        B1.config(state='disabled')
        [widget.config(state='enabled') for widget in [B_train_file, RB_alldata, val_ratio, RB_batch, RB_single, scale,
                                                       archi, epochs_entry, lr_entry, mm_entry, val_break_entry,
                                                       B_train, B_retrain, B_save, B2]]
        window5.get_tk_widget().grid_forget()
        B_test_file.grid_forget()
        testfile.grid_forget()
        B_net_response.grid_forget()
        [win[0].get_tk_widget().grid(column=win[1], row=win[2], padx=0, pady=0) for win in [(window1, 0, 1),
                                                                                            (window2, 1, 1),
                                                                                            (window3, 1, 2),
                                                                                            (window4, 0, 2)]]
        ttl_test.grid_forget()
        ttl_train.grid(column=0, columnspan=2, row=0, pady=5)

    def testing_win():
        B1.config(state='enabled')
        [button.config(state='disabled') for button in [B_train_file, RB_alldata, val_ratio, RB_batch, RB_single, scale,
                                                        archi, epochs_entry, lr_entry, mm_entry, val_break_entry, batch,
                                                        B_train, B_retrain, B_save, B2]]
        [win.get_tk_widget().grid_forget() for win in [window1, window2, window3, window4]]
        window5.get_tk_widget().grid(column=0, columnspan=2, row=1, padx=0, pady=0)
        B_test_file.grid(column=0, row=2, sticky='w,e', pady=5)
        testfile.grid(column=1, row=2, sticky='w,e', pady=5)
        B_net_response.grid(column=0, columnspan=2, row=3, sticky='w,e', pady=5)
        ttl_train.grid_forget()
        ttl_test.grid(column=0, columnspan=2, row=0, pady=5)

    # plotting figures

    matplotlib.rcParams['xtick.labelsize'] = 'small'
    matplotlib.rcParams['ytick.labelsize'] = 'small'

    figure1 = Figure(figsize=(6, 3.65), dpi=100, layout='tight')
    window1 = FigureCanvasTkAgg(figure1, rightframe)
    window1.get_tk_widget().grid(column=0, row=1, padx=0, pady=0)

    figure2 = Figure(figsize=(6, 3.65), dpi=100, layout='tight')
    window2 = FigureCanvasTkAgg(figure2, rightframe)
    window2.get_tk_widget().grid(column=1, row=1, padx=0, pady=0)

    figure3 = Figure(figsize=(6, 3.65), dpi=100, layout='tight')
    window3 = FigureCanvasTkAgg(figure3, rightframe)
    window3.get_tk_widget().grid(column=1, row=2, padx=0, pady=0)

    figure4 = Figure(figsize=(6, 3.65), dpi=100, layout='tight')
    window4 = FigureCanvasTkAgg(figure4, rightframe)
    window4.get_tk_widget().grid(column=0, row=2, padx=0, pady=0)

    figure5 = Figure(figsize=(12, 6.7), dpi=100, layout='tight')
    window5 = FigureCanvasTkAgg(figure5, rightframe)

    root.mainloop()
