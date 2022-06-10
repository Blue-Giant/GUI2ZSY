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
import DNN_Class_base


#  日志记数
def log_string(out_str, log_out):
    log_out.write(out_str + '\n')  # 将字符串写到文件log_fileout中去，末尾加换行
    log_out.flush()                # 清空缓存区
    # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
    # 一般情况下，文件关闭后会自动刷新缓冲区，但有时你需要在关闭前刷新它，这时就可以使用 flush() 方法。


class DNN2ZSY(object):
    def __init__(self, input_dim=1, out_dim=1, hidden_list=None, Model_name='DNN', actIn_name='tanh',
                 actHidden_name='tanh', actOut_name='linear', type2numeric='float32', freqs=None,
                 sFourier2Normal=1.0):
        super(DNN2ZSY, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN2Normal = DNN_Class_base.Pure_Dense_Net(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_list, name2Model=Model_name,
                actName2in=actIn_name, actName=actHidden_name, actName2out=actOut_name,
                type2float=type2numeric, scope2W='W_Normal', scope2B='B_Normal')
        elif 'SCALE_DNN' == str.upper(Model_name):
            self.DNN2Normal = DNN_Class_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_list, name2Model=Model_name,
                actName2in=actIn_name, actName=actHidden_name, actName2out=actOut_name,
                type2float=type2numeric, scope2W='W_Normal', scope2B='B_Normal', repeat_high_freq=False)
        elif 'FOURIER_DNN' == str.upper(Model_name):
            self.DNN2Normal = DNN_Class_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_list, name2Model=Model_name,
                actName2in=actIn_name, actName=actHidden_name, actName2out=actOut_name,
                type2float=type2numeric, scope2W='W_Normal', scope2B='B_Normal', repeat_high_freq=False)


class MY_GUI(object):
    def __init__(self, init_window):
        super(MY_GUI, self).__init__()
        self.init_window = init_window

        self.init_window.title("中石油项目")                  # 窗口名
        self.init_window.geometry('400x400+10+10')          # 290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        # self.init_window.geometry('1068x681+10+10')
        self.init_window["bg"] = "blue"                     # 窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        self.init_window.attributes("-alpha", 0.75)          # 虚化，值越小虚化程度越高

    # 设置窗口
    def set_init_window(self):
        # 标签
        self.name2DNN_label = TK.Label(self.init_window, text="模型名称", width=8)
        self.name2DNN_label.grid(row=0, column=0)

        self.act2input_label = TK.Label(self.init_window, text="输入层激活函数")
        self.act2input_label.grid(row=1, column=0)

        self.act2hiddens_label = TK.Label(self.init_window, text="隐藏层激活函数")
        self.act2hiddens_label.grid(row=2, column=0)

        self.act2output_label = TK.Label(self.init_window, text="输出层激活函数")
        self.act2output_label.grid(row=3, column=0)

        self.hiddens_label = TK.Label(self.init_window, text="网络大小", width=10)
        self.hiddens_label.grid(row=5, column=0)

        self.dim2in_label = TK.Label(self.init_window, text="输入数据维数")
        self.dim2in_label.grid(row=7, column=0)

        self.dim2out_label = TK.Label(self.init_window, text="输出维数")
        self.dim2out_label.grid(row=9, column=0)

        self.maxEpoch_label = TK.Label(self.init_window, text="最大迭代轮数")
        self.maxEpoch_label.grid(row=11, column=0)

        self.batchSize_label = TK.Label(self.init_window, text="批量大小")
        self.batchSize_label.grid(row=13, column=0)

        self.learning_rate_label = TK.Label(self.init_window, text="初始学习率")
        self.learning_rate_label.grid(row=15, column=0)

        self.Decay2lr_label = TK.Label(self.init_window, text="学习率衰减")
        self.Decay2lr_label.grid(row=17, column=0)

        # 文本框
        self.str2DNN = TK.StringVar(self.init_window)
        self.str2DNN.set('DNN')              # 默认 DNN
        self.name2DNN_Text = TK.OptionMenu(self.init_window, self.str2DNN, 'DNN', 'MscaleDNN', 'FourierDNN')  # 网络模型选择框
        self.name2DNN_Text.grid(row=0, column=2)

        str_list2act = ['ReLU', 'Leaky_ReLU', 'ELU', 'softplus', 'Tanh', 'Sin', 'sinAddcos', 's2ReLU', 'sReLU', 'mish', 'GELU', 'gauss', 'gcu', 'linear']
        self.str2act_In = TK.StringVar(self.init_window)
        self.str2act_In.set('ReLU')               # 默认 ReLU
        self.actIn_Text = TK.OptionMenu(self.init_window, self.str2act_In, *str_list2act)  # 输入层的激活函数
        self.actIn_Text.grid(row=1, column=2)

        self.str2act_Hidden = TK.StringVar()
        self.str2act_Hidden.set('ReLU')      # 默认 ReLU
        self.actHidden_Text = TK.OptionMenu(self.init_window, self.str2act_Hidden, *str_list2act)  # 隐藏层的激活函数
        self.actHidden_Text.grid(row=2, column=2)

        self.str2act_Out = TK.StringVar()
        self.str2act_Out.set('linear')       # 默认 linear
        self.actOut_Text = TK.OptionMenu(self.init_window, self.str2act_Out, *str_list2act)  # 输出层的激活函数
        self.actOut_Text.grid(row=3, column=2)

        self.str2Hidden = TK.StringVar()
        self.str2Hidden.set([200, 500, 100, 20])  # 是一个列表或者元组
        self.hiddens_Text = TK.Entry(self.init_window, width=20, textvariable=self.str2Hidden)  # 隐藏层神经单元列表输入框
        self.hiddens_Text.grid(row=5, column=2)

        self.int2in_dim = TK.IntVar()
        self.int2in_dim.set(2)              # 默认 2
        self.dim2in_Text = TK.Entry(self.init_window, width=20, textvariable=self.int2in_dim)  # 输入维度
        self.dim2in_Text.grid(row=7, column=2)

        self.int2out_dim = TK.IntVar()
        self.int2out_dim.set(1)             # 默认 1
        self.dim2out_Text = TK.Entry(self.init_window, width=20, textvariable=self.int2out_dim)  # 输出维度
        self.dim2out_Text.grid(row=9, column=2)

        int2Max_Epoch = TK.IntVar()
        int2Max_Epoch.set(10000)      # 默认 10000
        self.maxEpoch_Text = TK.Entry(self.init_window, width=20, textvariable=int2Max_Epoch)  # 最大迭代次数
        self.maxEpoch_Text.grid(row=11, column=2)

        int2BatchSize = TK.IntVar()
        int2BatchSize.set(16)      # 默认 10000
        self.BatchSize_Text = TK.Entry(self.init_window, width=20, textvariable=int2BatchSize)  # 批量大小
        self.BatchSize_Text.grid(row=13, column=2)

        double2lr = TK.DoubleVar()
        double2lr.set(0.01)  # 默认 10000
        self.learning_rate_Text = TK.Entry(self.init_window, width=20, textvariable=double2lr)  # 学习率
        self.learning_rate_Text.grid(row=15, column=2)

        double2lr_decay = TK.DoubleVar()
        double2lr_decay.set(0.001)  # 默认 10000
        self.decay2lr_Text = TK.Entry(self.init_window, width=20, textvariable=double2lr_decay)  # 学习率衰减
        self.decay2lr_Text.grid(row=17, column=2)

        self.button2Init = TK.Button(self.init_window, text='初始化参数', command=self.get_print_log_parameters)
        self.button2Init.grid(row=19, column=2)

        self.button2Start = TK.Button(self.init_window, text='开始训练', command=self.train_model)
        self.button2Start.grid(row=22, column=2)

        # self.button2Start.grid(row=22, column=2, rowspan=25, columnspan=5)
        # columnspan选项可以指定控件跨越多列显示，而rowspan选项同样可以指定控件跨越多行显示。

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

        name2model = self.str2DNN.get()
        log_string('Name for Network model: %s\n' % str(name2model), log_fileout)

        if name2model == 'Fourier_DNN' or name2model == 'FourierDNN':
            log_string('Input activate function for network: %s\n' % '[Sin;Cos]', log_fileout)
        else:
            log_string('Input activate function for network: %s\n' % str(self.str2act_In.get()), log_fileout)
        log_string('Hidden activate function for network: %s\n' % str(self.str2act_Hidden.get()), log_fileout)
        log_string('Output activate function for network: %s\n' % str(self.str2act_Out.get()), log_fileout)

        log_string('Hidden layers for model:%s\n' % str(self.str2Hidden.get()), log_fileout)

        log_string('The dim for input: %s\n' % str(self.dim2in_Text.get()), log_fileout)

        log_string('The dim for output: %s\n' % str(self.dim2out_Text.get()), log_fileout)

        log_string('Init learning rate: %s\n' % str(self.learning_rate_Text.get()), log_fileout)
        log_string('Decay to learning rate: %s\n' % str(self.decay2lr_Text.get()), log_fileout)

        log_string('Max epoch: %s\n' % str(self.maxEpoch_Text.get()), log_fileout)

        log_string('Batch-size: %s\n' % str(self.BatchSize_Text.get()), log_fileout)

        record_list = []
        list2hidden = []
        hidden_str = self.str2Hidden.get()
        # hidden_tuple = tuple(hidden_str)
        # print("hidden_tuple:", hidden_tuple)

        for str2num in hidden_str:
            if str2num != ' ' and str2num != '(' and str2num != ')' and str2num != ',':
                record_list.append(str2num)
                # print('record_list:', record_list)
            elif str2num == ',':
                list2hidden.append(int(''.join(record_list)))
                record_list.clear()
            elif str2num == ')':
                list2hidden.append(int(''.join(record_list)))
                record_list.clear()
        record_list.clear()

        print(type(list2hidden))
        print('list2hidden:', list2hidden)

        self.DNN = DNN2ZSY(input_dim=self.int2in_dim.get(), out_dim=self.int2out_dim.get(),
                           hidden_list=list2hidden, Model_name=name2model, actIn_name=self.str2act_In.get(),
                           actHidden_name=self.str2act_Hidden.get(), actOut_name=self.str2act_Out.get(),
                           type2numeric='float32',
                           freqs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                           sFourier2Normal=1.0)

    def train_model(self):
        print('Train model')
        global_steps = tf.compat.v1.Variable(0, trainable=False)
        with tf.device('/gpu:%s' % (0)):
            with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
                X_it = tf.compat.v1.placeholder(tf.float32, name='X_it', shape=[None, self.dim2in_Text.get()])  # * 行 1 列
                in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')
                print('------Train model------')

    def evalue_model(self):
        print('Test model')


if __name__ == '__main__':
    init_window = TK.Tk()  # 实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()  # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示