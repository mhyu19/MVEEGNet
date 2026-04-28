import torch
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.data import Batch
from loss import get_loss
from utils import add_gaussian_noise


def trainWithNoise(model, train_loader, target_loader, crit, optimizer, epoch, args):
    model.train()
    noise_level = 3.0 # 逐渐增强的噪声强度
    target_view = 3  # 对第4个视图加噪
    
    results = {
        'noise': noise_level,
        'accuracy': [],
        'u_target': [], # 受损视图的不确定性
        'u_others': []  # 正常视图的平均不确定性
    }
    
    loss_all = 0
    dynamic_loader = zip(train_loader, target_loader)
    for _, (source_data, target_data) in enumerate(dynamic_loader):
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
        optimizer.zero_grad()

        label = torch.argmax(source_data.y.view(-1, args.classes), dim=1) 
        # 注入噪声
        noisy_data = add_gaussian_noise(source_data, target_view, noise_level) 
        # import pdb; pdb.set_trace()
        class_output, _, smooth_loss = model(noisy_data)

        # multi_loss = get_loss(evs, ev_a, label, epoch, args.classes, 10, 0.1, args.device)
        cls_loss = crit(class_output, label)
        alpha = 2 / (1 + math.exp(-10 * epoch / 200)) - 1
        beta = alpha / 100
        total_loss = cls_loss +  beta * smooth_loss
        # total_loss = multi_loss + beta * smooth_loss

        total_loss.backward() 
        optimizer.step()  

        loss_all += total_loss.item() * source_data.num_graphs
    return (loss_all / len(train_loader.dataset))


def train(model, train_loader, target_loader, crit, optimizer, epoch, args):
    model.train()
    loss_all = 0
    dynamic_loader = zip(train_loader, target_loader)
    for _, (source_data, target_data) in enumerate(dynamic_loader):
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
        optimizer.zero_grad()

        label = torch.argmax(source_data.y.view(-1, args.classes), dim=1)  
        evs, ev_a, smooth_loss = model(source_data)

        multi_loss = get_loss(evs, ev_a, label, epoch, args.classes, 10, 0.1, args.device)
        # cls_loss = crit(class_output, label)
        alpha = 2 / (1 + math.exp(-10 * epoch / 200)) - 1
        beta = alpha / 100
        # total_loss = cls_loss + alpha * multi_loss + beta * smooth_loss
        total_loss = multi_loss + beta * smooth_loss

        total_loss.backward() 
        optimizer.step()  

        loss_all += total_loss.item() * source_data.num_graphs
    return (loss_all / len(train_loader.dataset))

def evaluate(model, loader, args):
    model.eval()
    predictions = []
    labels = []  
    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, args.classes)
            data = data.to(args.device)
            _, pred ,_ = model(data)
            # import pdb; pdb.set_trace()
            predictions.append(pred.cpu().detach().numpy())
            labels.append(label.numpy())
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=-1))
    return AUC, acc, f1, predictions, labels


def train2(model, train_loader, target_loader, crit, optimizer, epoch, args):
    model.train()
    loss_all = 0
    dynamic_loader = zip(train_loader, target_loader)
    for _, (source_data, target_data) in enumerate(dynamic_loader):
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
        optimizer.zero_grad()

        label = torch.argmax(source_data.y.view(-1, args.classes), dim=1)  
        class_output, _, smooth_loss = model(source_data)

        # multi_loss = get_loss(evs, ev_a, label, epoch, args.classes, 10, 0.1, args.device)
        cls_loss = crit(class_output, label)
        alpha = 2 / (1 + math.exp(-10 * epoch / 200)) - 1
        beta = alpha / 100
        total_loss = cls_loss +  beta * smooth_loss

        total_loss.backward() 
        optimizer.step()  

        loss_all += total_loss.item() * source_data.num_graphs
    return (loss_all / len(train_loader.dataset))

def evaluate2(model, loader, args):
    model.eval()
    predictions = []
    labels = []  
    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, args.classes)
            data = data.to(args.device)
            _, pred ,_ = model(data)
            predictions.append(pred.cpu().detach().numpy())
            labels.append(label.numpy())
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=-1))
    return AUC, acc, f1, predictions, labels


def train3(model, train_loader, target_loader, crit, optimizer, epoch, args):
    model.train()
    loss_all = 0
    dynamic_loader = zip(train_loader, target_loader)
    for _, (source_data, target_data) in enumerate(dynamic_loader):
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
        optimizer.zero_grad()

        label = torch.argmax(source_data.y.view(-1, args.classes), dim=1)  
        evs, ev_a, class_output,_ = model(source_data)

        multi_loss = get_loss(evs, ev_a, label, epoch, args.classes, 10, 0.1, args.device)
        cls_loss = crit(class_output, label)
        alpha = 2 / (1 + math.exp(-10 * epoch / 200)) - 1
        beta = alpha / 100
        total_loss = cls_loss + alpha * multi_loss 

        total_loss.backward() 
        optimizer.step()  

        loss_all += total_loss.item() * source_data.num_graphs
    return (loss_all / len(train_loader.dataset))

def evaluate3(model, loader, args):
    model.eval()
    predictions = []
    labels = []  
    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, args.classes)
            data = data.to(args.device)
            _, _, _, pred = model(data)
            predictions.append(pred.cpu().detach().numpy())
            labels.append(label.numpy())
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=-1))
    return AUC, acc, f1, predictions, labels


def train4(model, train_loader, target_loader, crit, optimizer, epoch, args):
    model.train()
    loss_all = 0
    dynamic_loader = zip(train_loader, target_loader)
    for _, (source_data, target_data) in enumerate(dynamic_loader):
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
        optimizer.zero_grad()

        label = torch.argmax(source_data.y.view(-1, args.classes), dim=1)  
        class_output, _, = model(source_data)

        # multi_loss = get_loss(evs, ev_a, label, epoch, args.classes, 10, 0.1, args.device)
        cls_loss = crit(class_output, label)
        # alpha = 2 / (1 + math.exp(-10 * epoch / 200)) - 1
        # beta = alpha / 100
        total_loss = cls_loss #+ alpha * multi_loss + beta * smooth_loss

        total_loss.backward() 
        optimizer.step()  

        loss_all += total_loss.item() * source_data.num_graphs
    return (loss_all / len(train_loader.dataset))

def evaluate4(model, loader, args):
    model.eval()
    predictions = []
    labels = []  
    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, args.classes)
            data = data.to(args.device)
            _, pred = model(data)
            predictions.append(pred.cpu().detach().numpy())
            labels.append(label.numpy())
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=-1))
    return AUC, acc, f1, predictions, labels