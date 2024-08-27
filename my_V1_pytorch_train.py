import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from tqdm.auto import tqdm


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data1, data2, data3, labels):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        # self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.data1[idx], self.data2[idx], self.data3[idx]), self.labels[idx]

class CustomDataset_pred(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        # self.data = data

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.data3[idx]
# 加载数据
def load_pred_data(X_for_pred):
    X_for_pred[0] = np.transpose(X_for_pred[0], (0, 3, 1, 2))
    X_for_pred[1] = np.transpose(X_for_pred[1], (0, 3, 1, 2))
    X_for_pred[2] = np.transpose(X_for_pred[2], (0, 3, 1, 2))
    # X_for_pred = np.transpose(X_for_pred, (0, 3, 1, 2))
    dataset_for_pred = CustomDataset_pred(X_for_pred[0], X_for_pred[1], X_for_pred[2])
    # dataset_for_pred = CustomDataset_pred(X_for_pred)
    return dataset_for_pred


def load_spot_recog_pred_data(X_test):
    dataset_for_pred = CustomDataset_pred(X_test[0], X_test[1], X_test[2])
    return dataset_for_pred


def load_data(X_train, X_test, y_train, y_test):
    X_train[0] = np.transpose(X_train[0], (0, 3, 1, 2))
    X_train[1] = np.transpose(X_train[1], (0, 3, 1, 2))
    X_train[2] = np.transpose(X_train[2], (0, 3, 1, 2))

    X_test[0] = np.transpose(X_test[0], (0, 3, 1, 2))
    X_test[1] = np.transpose(X_test[1], (0, 3, 1, 2))
    X_test[2] = np.transpose(X_test[2], (0, 3, 1, 2))
    train_dataset = CustomDataset(X_train[0], X_train[1], X_train[2], y_train)
    test_dataset = CustomDataset(X_test[0], X_test[1], X_test[2], y_test)
    # X_train = np.transpose(X_train, (0, 3, 1, 2))
    # X_test = np.transpose(X_test, (0, 3, 1, 2))
    # train_dataset = CustomDataset(X_train, y_train)
    # test_dataset = CustomDataset(X_test, y_test)
    return train_dataset, test_dataset

# 定义训练函数
def spot_train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, model_path='model.pth'):
    best_val_loss = float('inf')
    history = {'train_loss':[],'train_mae':[],'val_loss':[],'val_mae':[]}
    global_step = 0
    max_train_steps = len(train_loader) * num_epochs
    progress_bar = tqdm(range(global_step, max_train_steps), disable=False, position=0, leave=True)
    for epoch in range(num_epochs):
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        for i, ((inputs1, inputs2, inputs3), labels) in enumerate(train_loader):
            # # 确保输入张量是浮点数类型
            inputs1 = inputs1.float().cuda()
            inputs2 = inputs2.float().cuda()
            inputs3 = inputs3.float().cuda()
            # # 确保目标张量是浮点数类型
            # labels = labels.float()
            # inputs1, inputs2, inputs3, labels = inputs1.cuda(), inputs2.cuda(), inputs3.cuda(), labels.cuda()
            # inputs = inputs.float().cuda()
            labels = labels.float().cuda()

            # 零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs1, inputs2, inputs3)

            # 调整目标张量尺寸与输出张量一致
            labels = labels.view(-1, 1)

            loss = criterion(outputs, labels)

            # 计算MAE(MeanAbsoluteError)
            mae = torch.mean(torch.abs(outputs - labels)).item()


            # 反向传播
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            running_loss += loss.item()
            running_mae += mae

        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        train_mae = running_mae / len(train_loader)
        history['train_mae'].append(train_mae)

        # 验证
        val_loss, val_mae = validate_spot(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f},Train MAE: {running_mae/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val MAE:{val_mae:.4f}')

        # 保存最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with val loss: {best_val_loss:.4f}')
    return history

# 定义验证函数
def validate_spot(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        print("Validating...")
        # progress_bar = tqdm(range(0, len(val_loader)), leave=True, position=0, disable=False)
        for i, ((inputs1, inputs2, inputs3), labels) in enumerate(val_loader):
            # 确保输入张量是浮点数类型
            inputs1 = inputs1.float().cuda()
            inputs2 = inputs2.float().cuda()
            inputs3 = inputs3.float().cuda()
            # # 确保目标张量是浮点数类型
            # labels = labels.float()
            # inputs1, inputs2, inputs3, labels = inputs1.cuda(), inputs2.cuda(), inputs3.cuda(), labels.cuda()
            # inputs = inputs.float().cuda()
            labels = labels.float().cuda()

            # 前向传播
            outputs = model(inputs1, inputs2, inputs3)

            # 调整目标张量尺寸与输出张量一致
            labels = labels.view(-1, 1)

            loss = criterion(outputs, labels)
            # 计算MAE
            mae = torch.mean(torch.abs(outputs - labels)).item()
            # progress_bar.update(1)
            # logs = {"step_mae": mae}
            # progress_bar.set_postfix(**logs)
            val_loss += loss.item()
            val_mae += mae

    return val_loss / len(val_loader), val_mae / len(val_loader)

def recog_train(model, train_loader, test_loader, criterion, optimizer, num_epochs, model_path):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    global_step = 0
    max_train_steps = len(train_loader) * num_epochs
    progress_bar = tqdm(range(global_step, max_train_steps), disable=False, position=0, leave=True)
    for epoch in range(num_epochs):
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, ((inputs1, inputs2, inputs3), labels) in enumerate(train_loader):
            # 确保输入张量是浮点数类型
            inputs1 = inputs1.float().cuda()
            inputs2 = inputs2.float().cuda()
            inputs3 = inputs3.float().cuda()
            # # 确保目标张量是浮点数类型
            # labels = labels.float()
            # inputs1, inputs2, inputs3, labels = inputs1.cuda(), inputs2.cuda(), inputs3.cuda(), labels.cuda()
            # inputs = inputs.float().cuda()
            labels = labels.float().cuda()


            # 零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs1, inputs2, inputs3)

            labels_indice = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels_indice)

            # 计算准确率
            acc = calculate_accuracy(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            running_loss += loss.item()
            running_acc += acc

        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        train_acc = running_acc / len(train_loader)
        history['train_acc'].append(train_acc)

        # 验证
        val_loss, val_acc = validate_recog(model, test_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {running_acc/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 保存最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with val loss: {best_val_loss:.4f}')
    return history


def validate_recog(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        print("Validating...")
        # progress_bar = tqdm(range(len(val_loader)), leave=True, position=0, disable=False)
        for (inputs1, inputs2, inputs3), labels in val_loader:
            # # 确保输入张量是浮点数类型
            inputs1 = inputs1.float().cuda()
            inputs2 = inputs2.float().cuda()
            inputs3 = inputs3.float().cuda()
            # # 确保目标张量是浮点数类型
            # labels = labels.float()
            # inputs1, inputs2, inputs3, labels = inputs1.cuda(), inputs2.cuda(), inputs3.cuda(), labels.cuda()
            # inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            # 将 one-hot 编码的标签转换为类别索引
            labels_indices = torch.argmax(labels, dim=1)
            outputs = model(inputs1, inputs2, inputs3)

            loss = criterion(outputs, labels_indices)
            acc = calculate_accuracy(outputs, labels)
            # progress_bar.update(1)
            # logs = {"step_acc": acc}
            # progress_bar.set_postfix(**logs)
            running_loss += loss.item()
            running_acc += acc
    val_loss = running_loss / len(val_loader)
    val_acc = running_acc / len(val_loader)
    return val_loss, val_acc

# 定义准确率计算函数
def calculate_accuracy(outputs, labels):
    """
    计算准确率

    参数:
    outputs (torch.Tensor): 模型输出，形状为 (N, C)，其中 N 是批量大小，C 是类别数量
    labels (torch.Tensor): 真实标签，形状为 (N, C)，为 one-hot 编码

    返回:
    float: 准确率
    """
    # 将输出转换为预测的标签
    predicted_labels = torch.argmax(outputs, dim=1)

    # 将 one-hot 编码的标签转换为类别索引
    true_labels = torch.argmax(labels, dim=1)

    # 计算预测正确的数量
    correct = (predicted_labels == true_labels).sum().item()

    # 计算准确率
    accuracy = correct / labels.size(0)

    return accuracy



# # 主函数
# if __name__ == '__main__':
#     # 假设你的数据已经加载为 numpy 数组
#     # X_train, X_test = [np.array(X_train)[:, 0], np.array(X_train)[:, 1], np.array(X_train)[:, 2]], [
#     #     np.array(X_test)[:, 0], np.array(X_test)[:, 1], np.array(X_test)[:, 2]]
#     # y_train, y_test = np.array(y_train), np.array(y_test)
#
#     # 使用随机数据作为示例
#     X_train = [np.random.rand(100, 1, 42, 42).astype(np.float32) for _ in range(3)]
#     X_test = [np.random.rand(20, 1, 42, 42).astype(np.float32) for _ in range(3)]
#     y_train = np.random.rand(100, 1).astype(np.float32)
#     y_test = np.random.rand(20, 1).astype(np.float32)
#
#     # 加载数据
#     train_dataset, test_dataset = load_data(X_train, X_test, y_train, y_test)
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
#
#     # 实例化模型
#     model = torch_MEAN_Spot().cuda()
#
#     # 定义损失函数和优化器
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0005)
#
#     # 训练模型并验证
#     spot_train(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, model_path='best_model.pth')
#
#     # 加载最好的模型
#     model.load_state_dict(torch.load('best_model.pth'))
#     print("Best model loaded.")
