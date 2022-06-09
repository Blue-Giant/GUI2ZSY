#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as TK
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time


#  日志记数
def log_string(out_str, log_out):
    log_out.write(out_str + '\n')  # 将字符串写到文件log_fileout中去，末尾加换行
    log_out.flush()                # 清空缓存区
    # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
    # 一般情况下，文件关闭后会自动刷新缓冲区，但有时你需要在关闭前刷新它，这时就可以使用 flush() 方法。


class MY_GUI(object):
    def __init__(self, init_window_name):
        super(MY_GUI, self).__init__()
        self.init_window_name = init_window_name

        self.init_window_name.title("文本处理工具_v1.2")      # 窗口名
        self.init_window_name.geometry('400x500+10+10')     # 290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        # self.init_window_name.geometry('1068x681+10+10')
        self.init_window_name["bg"] = "blue"                # 窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        # self.init_window_name.attributes("-alpha",0.9)    # 虚化，值越小虚化程度越高

    # 设置窗口
    def set_init_window(self):
        # 标签
        self.name2DNN_label = TK.Label(self.init_window_name, text="模型名称", width=8)
        self.name2DNN_label.grid(row=0, column=0)

        self.hiddens_label = TK.Label(self.init_window_name, text="网络大小", width=10)
        self.hiddens_label.grid(row=1, column=0)

        self.act2input_label = TK.Label(self.init_window_name, text="输入层激活函数")
        self.act2input_label.grid(row=2, column=0)

        self.act2hiddens_label = TK.Label(self.init_window_name, text="隐藏层激活函数")
        self.act2hiddens_label.grid(row=3, column=0)

        self.act2output_label = TK.Label(self.init_window_name, text="输出层激活函数")
        self.act2output_label.grid(row=4, column=0)

        self.dim2in_label = TK.Label(self.init_window_name, text="输入数据维数")
        self.dim2in_label.grid(row=5, column=0)

        self.dim2out_label = TK.Label(self.init_window_name, text="输出维数")
        self.dim2out_label.grid(row=6, column=0)

        self.maxEpoch_label = TK.Label(self.init_window_name, text="最大迭代轮数")
        self.maxEpoch_label.grid(row=7, column=0)

        self.batchSize_label = TK.Label(self.init_window_name, text="批量大小")
        self.batchSize_label.grid(row=8, column=0)

        self.learning_rate_label = TK.Label(self.init_window_name, text="初始学习率")
        self.learning_rate_label.grid(row=9, column=0)

        self.Decay2lr_label = TK.Label(self.init_window_name, text="学习率衰减")
        self.Decay2lr_label.grid(row=10, column=0)

        # 文本框
        str2DNN = TK.StringVar()
        str2DNN.set('DNN')              # 默认 DNN
        self.name2DNN_Text = TK.Entry(self.init_window_name, width=20, textvariable=str2DNN)  # 网络模型选择框
        self.name2DNN_Text.grid(row=0, column=2)

        str2Hidden = TK.StringVar()
        str2Hidden.set('(5,10,20,5)')   # 是一个列表或者元组
        self.hiddens_Text = TK.Entry(self.init_window_name, width=20, textvariable=str2Hidden)    # 隐藏层神经单元列表输入框
        self.hiddens_Text.grid(row=1, column=2)

        str2act_In = TK.StringVar()
        str2act_In.set('ReLU')          # 默认 ReLU
        self.actIn_Text = TK.Entry(self.init_window_name, width=20, textvariable=str2act_In)  # 输入层的激活函数
        self.actIn_Text.grid(row=2, column=2)

        str2act_Hidden = TK.StringVar()
        str2act_Hidden.set('ReLU')      # 默认 ReLU
        self.actHidden_Text = TK.Entry(self.init_window_name, width=20, textvariable=str2act_Hidden)  # 隐藏层的激活函数
        self.actHidden_Text.grid(row=3, column=2)

        str2act_Out = TK.StringVar()
        str2act_Out.set('linear')       # 默认 linear
        self.actOut_Text = TK.Entry(self.init_window_name, width=20, textvariable=str2act_Out)  # 输出层的激活函数
        self.actOut_Text.grid(row=4, column=2)

        int2in_dim = TK.IntVar()
        int2in_dim.set(2)              # 默认 2
        self.dim2in_Text = TK.Entry(self.init_window_name, width=20, textvariable=int2in_dim)  # 输入维度
        self.dim2in_Text.grid(row=5, column=2)

        int2out_dim = TK.IntVar()
        int2out_dim.set(1)             # 默认 1
        self.dim2out_Text = TK.Entry(self.init_window_name, width=20, textvariable=int2out_dim)  # 输出维度
        self.dim2out_Text.grid(row=6, column=2)

        int2Max_Epoch = TK.IntVar()
        int2Max_Epoch.set(10000)      # 默认 10000
        self.maxEpoch_Text = TK.Entry(self.init_window_name, width=20, textvariable=int2Max_Epoch)  # 最大迭代次数
        self.maxEpoch_Text.grid(row=7, column=2)

        int2BatchSize = TK.IntVar()
        int2BatchSize.set(16)      # 默认 10000
        self.BatchSize_Text = TK.Entry(self.init_window_name, width=20, textvariable=int2BatchSize)  # 批量大小
        self.BatchSize_Text.grid(row=8, column=2)

        double2lr = TK.DoubleVar()
        double2lr.set(0.01)  # 默认 10000
        self.learning_rate_Text = TK.Entry(self.init_window_name, width=20, textvariable=double2lr)  # 学习率
        self.learning_rate_Text.grid(row=9, column=2)

        double2lr_decay = TK.DoubleVar()
        double2lr_decay.set(0.001)  # 默认 10000
        self.decay2lr_Text = TK.Entry(self.init_window_name, width=20, textvariable=double2lr_decay)  # 学习率衰减
        self.decay2lr_Text.grid(row=10, column=2)

        self.button2Start = TK.Button(self.init_window_name, text='开始训练', command=self.get_print_log_parameters)
        self.button2Start.grid(row=15, column=2, rowspan=16, columnspan=5)

    def get_print_log_parameters(self):
        num2GPU = 0
        if platform.system() == 'Windows':
            os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (num2GPU)
        else:
            print('-------------------------------------- linux -----------------------------------------------')
            matplotlib.use('Agg')

            if tf.test.is_gpu_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        # ------------------------------------------- 文件保存路径设置 ----------------------------------------
        store_file = 'VMD'
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(BASE_DIR)
        OUT_DIR = os.path.join(BASE_DIR, store_file)
        if not os.path.exists(OUT_DIR):
            print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
            os.mkdir(OUT_DIR)

        seed_num = np.random.randint(1e5)
        seed_str = str(seed_num)  # int 型转为字符串型
        FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
        if not os.path.exists(FolderName):
            print('--------------------- FolderName -----------------:', FolderName)
            os.mkdir(FolderName)

        # ----------------------------------------  复制并保存当前文件 -----------------------------------------
        if platform.system() == 'Windows':
            tf.compat.v1.reset_default_graph()
            shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
        else:
            shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

        if not os.path.exists(FolderName):  # 判断路径是否已经存在
            os.mkdir(FolderName)  # 无 log_out_path 路径，创建一个 log_out_path 路径

        outfile_name1 = '%s%s.txt' % ('log2', 'train')
        log_fileout = open(os.path.join(FolderName, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

        log_string('Name for Network model: %s\n' % str(self.name2DNN_Text.get()), log_fileout)
        log_string('Hidden layers for model:%s\n' % str(self.hiddens_Text.get()), log_fileout)

        if self.name2DNN_Text.get() == 'Fourier_DNN':
            log_string('Input activate function for network: %s\n' % '[Sin;Cos]', log_fileout)
        else:
            log_string('Input activate function for network: %s\n' % str(self.actIn_Text.get()), log_fileout)
        log_string('Hidden activate function for network: %s\n' % str(self.actHidden_Text.get()), log_fileout)
        log_string('Output activate function for network: %s\n' % str(self.actOut_Text.get()), log_fileout)

        log_string('The dim for input: %s\n' % str(self.dim2in_Text.get()), log_fileout)

        log_string('The dim for output: %s\n' % str(self.dim2out_Text.get()), log_fileout)

        log_string('Init learning rate: %s\n' % str(self.learning_rate_Text.get()), log_fileout)
        log_string('Decay to learning rate: %s\n' % str(self.decay2lr_Text.get()), log_fileout)

        log_string('Max epoch: %s\n' % str(self.maxEpoch_Text.get()), log_fileout)

        log_string('Batch-size: %s\n' % str(self.BatchSize_Text.get()), log_fileout)


if __name__ == '__main__':
    init_window = TK.Tk()  # 实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()  # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示