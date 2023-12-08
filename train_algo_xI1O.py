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

import numpy as np
import tkinter as tk

def tansig(val):
    t = (np.exp(val) - np.exp(-val)) / (np.exp(val) + np.exp(-val))
    return t

def d_tansig(val):
    dt = (2 / (np.exp(val) + np.exp(-val))) ** 2
    return dt

def deep_train_gd(net, inputs, targets, val_data_need, epochs, val_break, lr, mm, textbox, single):
    #gradient descent algorithm to train the network

    def update_textbox(text, write):
        if write:
            textbox.config(state='normal')
            for entry in range(0, text.__len__()):
                textbox.insert(tk.INSERT, text[entry])
            textbox.insert(tk.INSERT, "\n")
            textbox.see(tk.END)
            textbox.config(state='disabled')

    if val_data_need == 'with_val':
        val_inputs = inputs[1]
        val_targets = targets[1]
        inputs = inputs[0]
        targets = targets[0]
        validate = 1
    else:
        val_inputs = inputs
        val_targets = targets
        validate = 0

    val_count = 0
    mse_epoch = []
    mse_val_epoch = []
    dw_ep = []
    dwn_prev = []
    dwb_prev = []
    for i in range(0, len(net.Wb)):
        dwn_prev.append(0)
        dwb_prev.append(0)
        dw_ep.append([])

    # begin training iterations
    for ep in range(1, epochs + 1):
        dwn = []
        dwb = []
        for i in range(0, len(net.Wb)):
            dwn.append(0)
            dwb.append(0)

        # going through all input data
        for i in range(inputs.shape[0]):
            x = inputs[i].reshape(inputs.shape[1], 1)
            t = targets[i].transpose()
            n = []
            y = [x]

            # forward propagation
            for j in range(len(net.Wb)):
                n.append(net.Wn[j] @ x + net.Wb[j])
                if j < (len(net.Wb) - 1):
                    x = tansig(n[j])
                else:
                    x = n[j]
                y.append(x)

            # backpropagation and weight updates
            s_front = -2 * (t - y[-1])
            dwn[-1] = dwn[-1] - lr * (y[-2] * s_front).transpose()
            dwb[-1] = dwb[-1] - lr * s_front
            for j in np.arange(len(net.Wb) - 2, -1, -1):
                d_mat = np.eye(n[j].size)
                for k in range(n[j].size):
                    d_mat[k, k] = d_tansig(n[j][k])
                s = d_mat @ (net.Wn[j + 1]).transpose() @ s_front
                dwn[j] = dwn[j] - lr * s @ y[j].transpose()
                dwb[j] = dwb[j] - lr * s
                s_front = s

        # averaging the weight updates
        for i in np.arange(len(net.Wb) - 1, -1, -1):
            dwn[i] = dwn[i] / inputs.shape[0]
            dwb[i] = dwb[i] / inputs.shape[0]
            dw_ep[i].append(np.mean(dwn[i]) + np.mean(dwb[i]))

            # weights update
            net.Wn[i] = net.Wn[i] + (1 - mm) * dwn[i] + mm * dwn_prev[i]
            net.Wb[i] = net.Wb[i] + (1 - mm) * dwb[i] + mm * dwb_prev[i]

        dwn_prev = dwn
        dwb_prev = dwb

        # validation error / training error if no validation datasets are provided
        mse_val = 0
        for i in range(val_inputs.shape[0]):
            x = val_inputs[i].reshape(val_inputs.shape[1], 1)
            t = val_targets[i].transpose()
            n = []
            y = [x]
            for j in range(len(net.Wb)):
                n.append(net.Wn[j] @ x + net.Wb[j])
                if j < (len(net.Wb) - 1):
                    x = tansig(n[j])
                else:
                    x = n[j]
                y.append(x)
            mse = (t - y[-1]) ** 2
            mse_val = mse_val + mse
        mse_val = mse_val / val_inputs.shape[0]
        mse_val_epoch.append(np.mean(mse_val))

        # unique training error if validation datasets are provided
        if validate:
            mse_avg = 0
            for i in range(inputs.shape[0]):
                x = inputs[i].reshape(inputs.shape[1], 1)
                t = targets[i].transpose()
                n = []
                y = [x]
                for j in range(len(net.Wb)):
                    n.append(net.Wn[j] @ x + net.Wb[j])
                    if j < (len(net.Wb) - 1):
                        x = tansig(n[j])
                    else:
                        x = n[j]
                    y.append(x)
                mse = (t - y[-1]) ** 2
                mse_avg = mse_avg + mse
            mse_avg = mse_avg / inputs.shape[0]
            mse_epoch.append(np.mean(mse_avg))
        else:
            mse_epoch = mse_val_epoch

        # break training condition: validation error increasing
        if (ep > 1) and validate:
            if mse_val_epoch[ep - 1] >= mse_val_epoch[ep - 2]:
                val_count = val_count + 1
            else:
                val_count = 0
            if val_count == val_break:
                update_textbox("Divergence break", single)
                break

        # break training condition: convergence of validation/training error
        if (ep > 0.5*epochs) and (mse_val_epoch[ep - 1] < mse_val_epoch[ep - 2]):
            conv_count = conv_count + 1
        else:
            conv_count = 0
        if (conv_count > 10) and (abs((mse_val_epoch[ep - 1] - mse_val_epoch[ep - 2]) / mse_val_epoch[ep - 2]) < 0.01):
            update_textbox("Convergence break", single)
            break

    update_textbox(["Terminated in epoch = ", ep], single)
    update_textbox(["Final normalized training error = ", str(round(mse_epoch[-1], 5))], single)
    if validate:
        update_textbox(["Final normalized validation error = ", str(round(mse_val_epoch[-1], 5))], single)
    net.E_train = mse_epoch
    net.E_val = mse_val_epoch
    net.dW = dw_ep

    return net
