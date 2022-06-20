#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: li Xi'an（李西安）
 Date: 2022 年 5 月 31 日
"""
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
import DNN_data
import saveData
import plotData


#  日志记数函数
def log_string(out_str, log_out):
    log_out.write(out_str + '\n')  # 将字符串写到文件log_fileout中去，末尾加换行
    log_out.flush()                # 清空缓存区
    # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
    # 一般情况下，文件关闭后会自动刷新缓冲区，但有时你需要在关闭前刷新它，这时就可以使用 flush() 方法。


# 打印并记录训练过程的结果
def print_and_log_train_one_epoch(i_epoch, run_time, learn_rate, pwb, loss_ynn_tmp, loss_tmp, train_mse_tmp,
                                  train_res_tmp, log_out=None):
    # 将运行结果打印出来
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %.10f' % learn_rate)
    print('weights and biases with  penalty: %.10f' % pwb)
    print('loss for training: %.10f' % loss_tmp)
    print('solution mean square error for training: %.10f' % train_mse_tmp)
    print('solution residual error for training: %.10f\n' % train_res_tmp)

    log_string('train epoch: %d,time: %.10f' % (i_epoch, run_time), log_out)
    log_string('learning rate: %.10f' % learn_rate, log_out)
    log_string('weights and biases with  penalty: %.10f' % pwb, log_out)
    log_string('loss for training: %.10f' % loss_tmp, log_out)
    log_string('solution mean square error for training: %.10f' % train_mse_tmp, log_out)
    log_string('solution residual error for training: %.10f\n' % train_res_tmp, log_out)


# 打印并记录测试结果
def print_and_log_test_one_epoch(mse2test, res2test, log_out=None):
    # 将运行结果打印出来
    print('mean square error of predict and real for testing: %.10f' % mse2test)
    print('residual error of predict and real for testing: %.10f\n' % res2test)

    log_string('mean square error of predict and real for testing: %.10f' % mse2test, log_out)
    log_string('residual error of predict and real for testing: %.10f\n\n' % res2test, log_out)


class MY_GUI(object):
    def __init__(self, init_window, type2numeric='float32'):
        super(MY_GUI, self).__init__()
        self.init_window = init_window

        self.init_window.title("中石油项目")                  # 窗口名
        self.init_window.geometry('400x400+10+10')          # 290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        # self.init_window.geometry('400x450+10+10')          # 290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        # self.init_window.geometry('1068x681+10+10')
        self.init_window["bg"] = "blue"                     # 窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        self.init_window.attributes("-alpha", 0.8)          # 虚化，值越小虚化程度越高
        self.type2numeric = type2numeric

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

        self.batchSize_label = TK.Label(self.init_window, text="训练批量大小")
        self.batchSize_label.grid(row=13, column=0)

        self.learning_rate_label = TK.Label(self.init_window, text="初始学习率")
        self.learning_rate_label.grid(row=15, column=0)

        self.Decay2lr_label = TK.Label(self.init_window, text="学习率衰减")
        self.Decay2lr_label.grid(row=17, column=0)

        self.path2data_label = TK.Label(self.init_window, text="训练数据路径")
        self.path2data_label.grid(row=19, column=0)

        # 文本框
        self.str2DNN = TK.StringVar(self.init_window)
        self.str2DNN.set('DNN')              # 默认 DNN
        self.name2DNN_Text = TK.OptionMenu(self.init_window, self.str2DNN, 'DNN', 'MscaleDNN', 'FourierDNN')  # 网络模型选择框
        self.name2DNN_Text.grid(row=0, column=2)

        str_list2act = ['ReLU', 'Leaky_ReLU', 'ELU', 'softplus', 'Tanh', 'Sin', 'sinAddcos', 's2ReLU', 'sReLU', 'mish',
                        'GELU', 'MGELU', 'gauss', 'gcu', 'linear']
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
        self.str2Hidden.set('(200, 500, 100, 20)')  # 是一个列表或者元组
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

        self.int2Max_Epoch = TK.IntVar()
        self.int2Max_Epoch.set(10000)      # 默认 10000
        self.maxEpoch_Text = TK.Entry(self.init_window, width=20, textvariable=self.int2Max_Epoch)  # 最大迭代次数
        self.maxEpoch_Text.grid(row=11, column=2)

        self.int2BatchSize = TK.IntVar()
        self.int2BatchSize.set(16)      # 默认 10000
        self.BatchSize_Text = TK.Entry(self.init_window, width=20, textvariable=self.int2BatchSize)  # 训练批量大小
        self.BatchSize_Text.grid(row=13, column=2)

        self.double2lr = TK.DoubleVar()
        self.double2lr.set(0.01)  # 默认 10000
        self.learning_rate_Text = TK.Entry(self.init_window, width=20, textvariable=self.double2lr)  # 学习率
        self.learning_rate_Text.grid(row=15, column=2)

        self.double2lr_decay = TK.DoubleVar()
        self.double2lr_decay.set(0.001)  # 默认 10000
        self.decay2lr_Text = TK.Entry(self.init_window, width=20, textvariable=self.double2lr_decay)  # 学习率衰减
        self.decay2lr_Text.grid(row=17, column=2)

        self.str_path2data = TK.StringVar()
        self.str_path2data.set('data/..')
        self.path2data_Text = TK.Entry(self.init_window, width=25, textvariable=self.str_path2data)  # 数据路径
        self.path2data_Text.grid(row=19, column=2)

        button2Init = TK.Button(self.init_window, text='初始化参数和网络', command=self.get_print_log_parameters)
        button2Init.grid(row=22, column=2)

        # type2loss = 'l2loss'
        type2loss = 'lnchshloss'
        test_bachsize=200
        button2Start = TK.Button(self.init_window, text='开始训练',
                                 command=lambda: self.train_test_model(loss_type=type2loss, batchsize_test=test_bachsize))
        button2Start.grid(row=25, column=2)

        # button2End = TK.Button(self.init_window, text='测试网络', command=self.evalue_model)
        # button2End.grid(row=28, column=2)

        # self.button2Start.grid(row=22, column=2, rowspan=25, columnspan=5)
        # columnspan选项可以指定控件跨越多列显示，而rowspan选项同样可以指定控件跨越多行显示。

    # 得到并初始化参数
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
        store_file = 'ZSY'
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(BASE_DIR)
        OUT_DIR = os.path.join(BASE_DIR, store_file)
        if not os.path.exists(OUT_DIR):
            print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
            os.mkdir(OUT_DIR)

        seed_num = np.random.randint(1e5)
        self.seed = str(seed_num)
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

        self.Name2Folder = FolderName

        outfile_name1 = '%s%s.txt' % ('log2', 'train')
        log_fileout = open(os.path.join(FolderName, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
        self.log_outfile = log_fileout

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

        data_path = self.str_path2data.get()
        print('data_path:', data_path)

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
                # print('record_list for douhao:', record_list)
                record_list.clear()
            elif str2num == ')':
                list2hidden.append(int(''.join(record_list)))
                # print('record_list for right ):', record_list)
                record_list.clear()
            elif str2num is None:
                list2hidden.append(int(''.join(record_list)))
                # print('record_list for end:', record_list)
                record_list.clear()
        record_list.clear()

        # print(type(list2hidden))
        # print('list2hidden:', list2hidden)

        if self.type2numeric == 'float32':
            self.float_type = tf.float32
        elif self.type2numeric == 'float64':
            self.float_type = tf.float64
        elif self.type2numeric == 'float16':
            self.float_type = tf.float16

        if 'DNN' == str.upper(name2model):
            self.DNN = DNN_Class_base.Pure_Dense_Net(
                indim=self.int2in_dim.get(), outdim=self.int2out_dim.get(), hidden_units=list2hidden, name2Model=name2model,
                actName2in=self.str2act_In.get(), actName=self.str2act_Hidden.get(), actName2out=self.str2act_Out.get(),
                type2float=self.type2numeric, scope2W='Ws', scope2B='Bs')
        elif 'SCALE_DNN' == str.upper(name2model) or 'MSCALEDNN' == str.upper(name2model):
            self.DNN = DNN_Class_base.Dense_ScaleNet(
                indim=self.int2in_dim.get(), outdim=self.int2out_dim.get(), hidden_units=list2hidden, name2Model=name2model,
                actName2in=self.str2act_In.get(), actName=self.str2act_Hidden.get(), actName2out=self.str2act_Out.get(),
                type2float=self.type2numeric, scope2W='Ws', scope2B='Bs', repeat_high_freq=False)
        elif 'FOURIER_DNN' == str.upper(name2model) or 'FOURIERDNN' == str.upper(name2model):
            self.DNN = DNN_Class_base.Dense_FourierNet(
                indim=self.int2in_dim.get(), outdim=self.int2out_dim.get(), hidden_units=list2hidden, name2Model=name2model,
                actName2in=self.str2act_In.get(), actName=self.str2act_Hidden.get(), actName2out=self.str2act_Out.get(),
                type2float=self.type2numeric, scope2W='Ws', scope2B='Bs', repeat_high_freq=False)

    # 训练并测试网络
    def train_test_model(self, loss_type='l2loss', batchsize_test=100):
        # print('Train model')
        # print('loss_type:', loss_type)
        # print('batchsize_test:', batchsize_test)
        freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        penalty2WB = 0.001
        global_steps = tf.compat.v1.Variable(0, trainable=False)
        with tf.device('/gpu:%s' % (0)):
            with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
                X_train = tf.compat.v1.placeholder(tf.float32, name='X_train', shape=[None, self.int2in_dim.get()])
                X_test = tf.compat.v1.placeholder(tf.float32, name='X_test', shape=[None, self.int2in_dim.get()])
                Y_label2train = tf.compat.v1.placeholder(tf.float32, name='Y_label2train', shape=[None, self.int2out_dim.get()])
                in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

                YNN_train = self.DNN(X_train, scale=freqs, sFourier=1.0)

                if loss_type == 'l2loss' or loss_type == 'l2_loss':
                    Loss2YNN = tf.reduce_mean(tf.square(YNN_train - Y_label2train))
                elif loss_type == 'lncosh_loss' or loss_type == 'lncoshloss':
                    Loss2YNN = tf.reduce_mean(tf.log(tf.cosh(YNN_train - Y_label2train)))
                else:
                    raise IndexError('No loss')

                regularSum2WB = self.DNN.get_regular_sum2WB(regular_model='L1')
                PWB = penalty2WB * regularSum2WB

                Loss2All = Loss2YNN + PWB

                my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
                train_Loss2NN = my_optimizer.minimize(Loss2All, global_step=global_steps)

                train_mse_NN = tf.reduce_mean(tf.square(YNN_train - Y_label2train))
                train_rel_NN = train_mse_NN / tf.reduce_mean(tf.square(Y_label2train))

                YNN_test = self.DNN(X_train, scale=freqs, sFourier=1.0)

        t0 = time.time()
        loss2y_all, loss_all, train_mse_all, train_rel_all = [], [], [], []
        test_mse_all, test_rel_all = [], []
        test_epoch = []

        test_x_bach = np.reshape(np.linspace(region_l, region_r, num=batchsize_test), [-1, 1])
        saveData.save_testData_or_solus2mat(test_x_bach, dataName='testx', outPath=self.Name2Folder)

        # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
        config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
        config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            tmp_lr = self.double2lr.get()

            for i_epoch in range(self.int2Max_Epoch.get() + 1):
                x_train_batch = DNN_data.rand_it(self.int2BatchSize.get(), self.int2in_dim.get(), region_a=region_l,
                                                 region_b=region_r)
                tmp_lr = tmp_lr * (1 - self.double2lr_decay.get())

                _, loss_ynn, loss_nn, train_mse_nn, train_rel_nn, pwb = sess.run(
                    [train_Loss2NN, Loss2All, Loss2YNN, train_mse_NN, train_rel_NN, PWB],
                    feed_dict={X_train: x_train_batch, Y_label2train: train_ylabel_batch, in_learning_rate: tmp_lr})
                loss2y_all.append(loss_ynn)
                loss_all.append(loss_nn)
                train_mse_all.append(train_mse_nn)
                train_rel_all.append(train_rel_nn)

                if i_epoch % 1000 == 0:
                    run_times = time.time() - t0
                    print_and_log_train_one_epoch(i_epoch, run_times, tmp_lr, pwb, loss_ynn, loss_nn, train_mse_nn,
                                                  train_rel_nn, log_out=self.log_outfile)

                    # ---------------------------   test network ----------------------------------------------
                    test_epoch.append(i_epoch / 1000)
                    ynn2test = sess.run([YNN_test], feed_dict={X_test: test_x_bach})
                    test_mse2nn = np.mean(np.square(ynn2test - test_ylabel_batch))
                    test_mse_all.append(test_mse2nn)
                    test_rel2nn = test_mse2nn / np.mean(np.square(test_ylabel_batch))
                    test_rel_all.append(test_rel2nn)

                    print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=self.log_outfile)

        # -----------------------  save training results to mat files, then plot them ---------------------------------
        saveData.save_trainLoss2mat(loss2y_all, loss_all, actName=self.str2act_Hidden.get(), outPath=self.log_outfile)

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=self.str2act_Hidden.get(), outPath=self.log_outfile)

        plotData.plotTrain_loss_1act_func(loss2y_all, lossType='loss_it', seedNo=self.seed, outPath=self.log_outfile,
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=self.seed, outPath=self.log_outfile,
                                          yaxis_scale=True)

        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=self.str2act_Hidden.get(),
                                             seedNo=self.seed, outPath=self.log_outfile, yaxis_scale=True)

        # ----------------------  save testing results to mat files, then plot them --------------------------------
        saveData.save_testData_or_solus2mat(test_ylabel_batch, dataName='Utrue', outPath=self.log_outfile)
        saveData.save_testData_or_solus2mat(ynn2test, dataName=self.str2act_Hidden.get(), outPath=self.log_outfile)
        plotData.plot_2solutions2test(test_ylabel_batch, ynn2test, coord_points2test=test_x_bach,
                                      batch_size2test=batchsize_test, seedNo=self.seed, outPath=self.log_outfile,
                                      subfig_type=0, scatter_fig=False, actName='test')

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=self.str2act_Hidden.get(), outPath=self.log_outfile)
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=self.str2act_Hidden.get(),
                                  seedNo=self.seed, outPath=self.log_outfile, yaxis_scale=True)

    def evalue_model(self, freqs=None):
        print('Test model')
        # freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        with tf.device('/gpu:%s' % (0)):
            with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
                X_test = tf.compat.v1.placeholder(tf.float32, name='X_test', shape=[None, self.int2in_dim.get()])
                y_pre = self.DNN(X_test, scale=freqs, sFourier=1.0)


if __name__ == '__main__':
    gui_window = TK.Tk()              # 实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(gui_window)   # 初始化窗口

    ZMJ_PORTAL.set_init_window()      # 设置根窗口默认属性

    gui_window.mainloop()             # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
