import copy
from functools import partial


import os

import torch
# from tensorflow import keras
import numpy as np
import random
# from collections import Counter, OrderedDict
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn, optim
from torch.utils.data import DataLoader


from model.AMEAN_BMOF import torch_AMEAN_Spot, torch_p_AMEAN_Spot, torch_AMEAN_Recog_TL, torch_AMEAN_Spot_Recog_TL, \
    torch_q_model_spot, torch_q_model_recog, load_s_r_dict
# from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
from sklearn.metrics import accuracy_score
import time

from my_V1_pytorch_train import load_data, spot_train, recog_train, load_pred_data, load_spot_recog_pred_data
from training_utils import *


random.seed(1)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def pytorch_train_test(X, y, X1, y1, X2, y2, dataset_name, emotion_class, groupsLabel, groupsLabel1, spot_multiple,
               final_subjects, final_emotions, final_samples, final_dataset_spotting, k, k_p, expression_type,
               epochs_spot=10, epochs_recog=100, spot_lr=0.01, recog_lr=0.01, batch_size=8, ratio=5, p=0.55,
               spot_attempt=1, recog_attempt=1, train_spot=False, train_recog=False):
    start = time.time()
    loso = LeaveOneGroupOut()
    subject_count = 0
    total_gt_spot = 0
    metric_final = MeanAveragePrecision2d(num_classes=1)

    # 构建检测网络
    torch_mean_spot = torch_MEAN_Spot().cuda()
    # weight_reset_spot = model_spot.get_weights()  # Initial weights
    # 初始化检测网络的权重，保证loso训练的统一初始化
    # initial_state_dict = copy.deepcopy(torch_mean_spot.state_dict())
    # 定义损失函数和优化器
    # 检测网络的MSE损失函数以及L2正则化weight_decay
    spot_criterion = nn.MSELoss()
    p_model_spot = torch_p_MEAN_Spot().cuda()
    q_model_spot = torch_q_model_spot().cuda()
    q_model_recog = torch_q_model_recog().cuda()

    print('---------------------------')

    # For Spotting
    gt_spot_list = []
    pred_spot_list = []
    # For recognition
    gt_list = []
    pred_list = []
    gt_tp_list = []
    pred_ori_list = []
    pred_window_list = []
    pred_single_list = []
    asr_score = 0
    # For LOSO
    spot_train_index = []
    spot_test_index = []
    recog_train_index = []
    recog_test_index = []

    recog_subject_uni = np.unique(groupsLabel1)  # Get unique subject label
    for train_index, test_index in loso.split(X, y, groupsLabel):  # Spotting Leave One Subject Out
        spot_train_index.append(train_index)
        spot_test_index.append(test_index)
    for subject_index, (train_index, test_index) in enumerate(
            loso.split(X1, y1, groupsLabel1)):  # Recognition Leave One Subject Out
        if (
                subject_index not in recog_subject_uni):  # To remove subject that don't have chosen emotions for evalaution
            recog_train_index.append(np.array([]))
            recog_test_index.append(np.array([]))
        recog_train_index.append(train_index)
        recog_test_index.append(test_index)

    for subject_index in range(len(final_subjects)):
        subject_count += 1
        print('Index: ' + str(subject_count - 1) + ' | Subject : ' + str(final_subjects[subject_count - 1]))
        torch_mean_spot = torch_MEAN_Spot().cuda()  # 重新实例化 Spotting 网络
        torch_model_recog = torch_MEAN_Recog_TL().cuda()  # 重新实例化 Recognition 网络

        # Prepare training & testing data by loso splitting
        X_train, X_test = [X[i] for i in spot_train_index[subject_index]], [X[i] for i in spot_test_index[
            subject_index]]  # Get training set spotting
        y_train, y_test = [y[i] for i in spot_train_index[subject_index]], [y[i] for i in spot_test_index[
            subject_index]]  # Get testing set spotting
        X1_train, X1_test = [X1[i] for i in recog_train_index[subject_index]], [X1[i] for i in recog_test_index[
            subject_index]]  # Get training set recognition
        y1_train, y1_test = [y1[i] for i in recog_train_index[subject_index]], [y1[i] for i in recog_test_index[
            subject_index]]  # Get testing set recognition
        X2_train, y2_train = [X2[i] for i in recog_train_index[subject_index]], [y2[i] for i in recog_train_index[
            subject_index]]  # Get training set recognition
        X2_test, y2_test = [X2[i] for i in recog_test_index[subject_index]], [y2[i] for i in recog_test_index[
            subject_index]]  # Get testing set recognition

        print('Dataset Labels (Spotting, Recognition)', Counter(y_train), Counter([argmax(i) for i in y1_train]))

        # Make the dataset in the expected ratio by randomly removing training samples
        unique, uni_count = np.unique(y_train, return_counts=True)
        rem_count = int(uni_count.min() * ratio)
        if (rem_count <= len(y_train)):
            rem_index = random.sample([index for index, i in enumerate(y_train) if i == 0], rem_count)
            rem_index += (index for index, i in enumerate(y_train) if i == 1)
        else:
            rem_count = int(uni_count.max() / ratio)
            rem_index = random.sample([index for index, i in enumerate(y_train) if i == 1], rem_count)
            rem_index += (index for index, i in enumerate(y_train) if i == 0)
        rem_index.sort()

        X_train = [X_train[i] for i in rem_index]
        y_train = [y_train[i] for i in rem_index]

        print('After Downsampling (Spotting, Recognition)', Counter(y_train), Counter([argmax(i) for i in y1_train]))

        print('------ MEAN Spotting-------')  # To reset the model at every LOSO testing
        # path = 'MEAN_Weights\\' + dataset_name + '\\' + 'spot' + '\\s' + str(subject_count) + '.hdf5'
        path = 'F:\\my_N_Weights\\' + dataset_name + '\\' + 'spot' + '\\s' + str(subject_count) + '.pth'

        # Prepare training & testing data
        X_train, X_test = [np.array(X_train)[:, 0], np.array(X_train)[:, 1], np.array(X_train)[:, 2]], [
            np.array(X_test)[:, 0], np.array(X_test)[:, 1], np.array(X_test)[:, 2]]
        # X_train, X_test = np.array(X_train), np.array(X_test)

        # 加载数据
        train_dataset, test_dataset = load_data(X_train, X_test, np.array(y_train), np.array(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if not train_spot:
            torch_mean_spot.load_state_dict(torch.load(path))
        else:
            # 训练检测网络
            # torch_mean_spot.load_state_dict(initial_state_dict)
            history_spot = spot_train(torch_mean_spot, train_loader, test_loader, spot_criterion, adam_spot, num_epochs=epochs_spot, model_path=path)


        # 加载通过检测网路训练的编码器权重到识别网路中
        model_static = copy.deepcopy(torch_mean_spot.state_dict())
        model_static.pop("interpretation.1.weight")
        model_static.pop("interpretation.1.bias")
        model_static.pop("interpretation.3.weight")
        model_static.pop("interpretation.3.bias")
        model_static.pop("interpretation.5.weight")
        model_static.pop("interpretation.5.bias")
        p_model_spot.load_state_dict(model_static, strict=True)
        # q_model_spot = torch_q_model_spot()

        # 为合并网络构建检测网络的解码器部分
        # q_model_spot = MAE_q_model_spot(decoder1)

        # 为合并网络构建识别网络的解码器部分
        # q_model_recog = MAE_q_model_recog(decoder2)


        # path = 'MEAN_Weights\\' + dataset_name + '\\' + 'recog' + '\\s' + str(subject_count) + '.hdf5'
        path = 'F:\\my_N_Weights\\' + dataset_name + '\\' + 'recog' + '\\s' + str(subject_count) + '.pth'
        # torch_model_recog = torch_MEAN_Recog_TL(encoder, emotion_class).cuda()
        torch_model_recog = torch_MEAN_Recog_TL(p_model_spot).cuda()
        # adam_recog = optim.Adam(torch_model_recog.parameters(), lr=recog_lr)
        # 识别网路的交叉熵损失函数以及L2正则化weight_decay
        recog_criterion = nn.CrossEntropyLoss()
        adam_recog = optim.Adam(torch_model_recog.parameters(), lr=recog_lr)
        adam_spot = optim.Adam(torch_mean_spot.parameters(), lr=0.0005)

        if (len(X1_train) > 0):  # Check the subject has samples for recognition
            print('------ MEAN Recognition-------')  # Using transfer learning for recognition
            # Prepare training & testing data
            X1_train, X1_test = [np.array(X1_train)[:, 0], np.array(X1_train)[:, 1], np.array(X1_train)[:, 2]], [
                np.array(X1_test)[:, 0], np.array(X1_test)[:, 1], np.array(X1_test)[:, 2]]
            X2_train, X2_test = [np.array(X2_train)[:, 0], np.array(X2_train)[:, 1], np.array(X2_train)[:, 2]], [
                np.array(X2_test)[:, 0], np.array(X2_test)[:, 1], np.array(X2_test)[:, 2]]
            # X1_train, X1_test = np.array(X1_train), np.array(X1_test)
            # X2_train, X2_test = np.array(X2_train), np.array(X2_test)

            # 加载数据
            train_dataset, test_dataset = load_data(X2_train, X1_test, np.array(y2_train), np.array(y1_test))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            if not train_recog:    # Load Pretrained Weights
                torch_model_recog.load_state_dict(torch.load(path))
            else:
                # 训练识别网络
                history_recog = recog_train(torch_model_recog, train_loader, test_loader, recog_criterion, adam_recog,
                                            num_epochs=epochs_recog, model_path=path)
            result_recog_ori = recog_predict_pytorch(torch_model_recog, X2_test, batch_size)


            if train_recog:  # Plot graph to see performance
                my_torch_history_plot(history_recog, str(final_subjects[subject_count - 1]))

        path = 'F:\\my_N_Weights\\' + dataset_name + '\\' + 'spot_recog' + '\\s' + str(subject_count) + '.pth'
        # model_spot_recog = torch_MEAN_Spot_Recog_TL(encoder, q_model_spot, q_model_recog)

        # 构建整体网络
        model_spot_recog = torch_MEAN_Spot_Recog_TL(p_model_spot, q_model_spot, q_model_recog)
        model_spot_recog_state_dict = load_s_r_dict(torch_mean_spot, torch_model_recog)
        # params = model_spot_recog.state_dict()
        # params.update(model_spot_recog_state_dict)
        model_spot_recog.load_state_dict(model_spot_recog_state_dict, strict=True)
        model_spot_recog.cuda()
        if train_recog:
            torch.save(model_spot_recog.state_dict(), path)  # Save Weights
        else:
            model_spot_recog.load_state_dict(torch.load(path))  # Load Pretrained Weights

        results = spot_recog_predict_pytorch(model_spot_recog, X_test, batch_size)

        print('---- Spotting Results ----')
        preds, gt, total_gt_spot, metric_video, metric_final = spotting(results[0], total_gt_spot, subject_count, p,
                                                                        metric_final, spot_multiple, k_p, final_samples,
                                                                        final_dataset_spotting)
        TP_spot, FP_spot, FN_spot = sequence_evaluation(total_gt_spot, metric_final)
        try:
            precision = TP_spot / (TP_spot + FP_spot)
            recall = TP_spot / (TP_spot + FN_spot)
            F1_score = (2 * precision * recall) / (precision + recall)
            # print('F1-Score = ', round(F1_score, 4))
            # print("COCO AP@[.5:.95]:", round(metric_final.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
        except:
            pass
        pred_spot_list.extend(preds)
        gt_spot_list.extend(gt)
        asr_score, mae_score = apex_evaluation(pred_spot_list, gt_spot_list, k_p)
        # print('ASR:', round(asr_score,4))
        # print('MAE:', round(mae_score,4))

        if (len(X1_train) > 0):  # Check the subject has samples for recognition
            # Recognition
            print('---- Recognition Results ----')
            gt_list.extend(list(argmax(y2_test, -1)))
            pred_ori_list.extend(list(argmax(result_recog_ori, -1)))
            pred_list, gt_tp_list, pred_window_list, pred_single_list = recognition(dataset_name, emotion_class,
                                                                                    results[1], preds, metric_video,
                                                                                    final_emotions, subject_count,
                                                                                    pred_list, gt_tp_list, y_test,
                                                                                    final_samples, pred_window_list,
                                                                                    pred_single_list, spot_multiple, k,
                                                                                    k_p, final_dataset_spotting)
            print('Ground Truth           :', list(argmax(y2_test, -1)))

    print('Done Index: ' + str(subject_count - 1) + ' | Subject : ' + str(final_subjects[subject_count - 1]))

    end = time.time()
    print('Total time taken for training & testing: ' + str(end - start) + 's')
    return TP_spot, FP_spot, FN_spot, metric_final, gt_list, pred_list, gt_tp_list, asr_score, mae_score


def recog_predict_pytorch(model, X2_test, batch_size):
    # 准备数据
    test_dataset = load_pred_data(X2_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型评估
    model.eval()  # 设置模型为评估模式
    result_recog_ori = []
    total_batches = len(test_loader)
    with torch.no_grad():  # 关闭梯度计算
        for i, (inputs1, inputs2, inputs3) in enumerate(test_loader):
            # 确保输入张量是浮点数类型
            inputs1 = inputs1.float().cuda()
            inputs2 = inputs2.float().cuda()
            inputs3 = inputs3.float().cuda()
            # inputs1, inputs2, inputs3 = inputs1.cuda(), inputs2.cuda(), inputs3.cuda()
            # input = input.float().cuda()
            outputs = model(inputs1, inputs2, inputs3)
            result_recog_ori.append(outputs)
            # print(f'Batch {i + 1}/{total_batches} completed')
    # 将结果拼接成一个数组
    result_recog_ori = torch.cat(result_recog_ori).cpu().numpy()
    return result_recog_ori


def spot_recog_predict_pytorch(model, X_test, batch_size):
    # 准备数据
    test_dataset = load_spot_recog_pred_data(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型评估
    model.eval()  # 设置模型为评估模式
    spot_results = []
    recog_results = []
    total_batches = len(test_loader)
    with torch.no_grad():  # 关闭梯度计算
        for i, (inputs1, inputs2, inputs3) in enumerate(test_loader):
            # # 确保输入张量是浮点数类型
            inputs1 = inputs1.float().cuda()
            inputs2 = inputs2.float().cuda()
            inputs3 = inputs3.float().cuda()
            # inputs1, inputs2, inputs3 = inputs1.cuda(), inputs2.cuda(), inputs3.cuda()
            # inputs = inputs.float().cuda()
            spot_output, recog_output = model(inputs1, inputs2, inputs3)
            spot_results.append(spot_output.cpu())
            recog_results.append(recog_output.cpu())
            # print(f'Batch {i + 1}/{total_batches} completed')
    # 将结果拼接成一个数组
    # 将结果列表转换为张量
    spot_results = torch.cat(spot_results, dim=0)
    recog_results = torch.cat(recog_results, dim=0)
    return spot_results, recog_results


def final_evaluation(TP_spot, FP_spot, FN_spot, dataset_name, expression_type, metric_final, asr_score, mae_score,
                     spot_multiple, pred_list, gt_list, emotion_class, gt_tp_list):
    # Spotting
    precision = TP_spot / (TP_spot + FP_spot)
    recall = TP_spot / (TP_spot + FN_spot)
    F1_score = (2 * precision * recall) / (precision + recall)
    print('----Spotting----')
    print('Final Result for', dataset_name)
    print('TP:', TP_spot, 'FP:', FP_spot, 'FN:', FN_spot)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:",
          round(metric_final.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
    print('ASR = ', round(asr_score, 4))
    print('MAE = ', round(mae_score, 4))

    # Check recognition accuracy if only correctly predicted spotting are considered
    if (not spot_multiple):
        print('\n----Recognition (All)----')
        print('Predicted    :', pred_list)
        print('Ground Truth :', gt_list)
        UF1, UAR = recognition_evaluation(dataset_name, emotion_class, gt_list, pred_list, show=True)
        print('Accuracy Score:', round(accuracy_score(gt_list, pred_list), 4))

    print('\n----Recognition (Consider TP only)----')
    gt_tp_spot = []
    pred_tp_spot = []
    for index in range(len(gt_tp_list)):
        if (gt_tp_list[index] != -1):
            gt_tp_spot.append(gt_tp_list[index])
            pred_tp_spot.append(pred_list[index])
    print('Predicted    :', pred_tp_spot)
    print('Ground Truth :', gt_tp_spot)
    UF1, UAR = recognition_evaluation(dataset_name, emotion_class, gt_tp_spot, pred_tp_spot, show=True)
    print('Accuracy Score:', round(accuracy_score(gt_tp_spot, pred_tp_spot), 4))
