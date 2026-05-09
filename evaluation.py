import torch
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.data import Batch
from loss import ce_loss


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

        multi_loss = 0 
        for v in range(len(evs)):
            multi_loss += ce_loss(label, evs[v], args.classes, epoch, 10, args.device)
        multi_loss += ce_loss(label, ev_a, args.classes, epoch, 10, args.device)
        alpha = 2 / (1 + math.exp(-10 * epoch / 200)) - 1
        beta = alpha / 100
        total_loss = multi_loss + beta * smooth_loss

        total_loss.backward() 
        optimizer.step()  

        loss_all += total_loss.item() * source_data.num_graphs
    return (loss_all / len(train_loader.dataset))

def evaluate(model, loader, args):
    model.eval()
    predictions = []
    labels = []  
    u_s = []
    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, args.classes)
            data = data.to(args.device)
            _, u_fused, pred, _,_ = model(data)
            # print(pred)
            predictions.append(pred.cpu().detach().numpy())
            labels.append(label.numpy())
            u_s.append(u_fused.cpu().detach().numpy())
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    u_s = np.vstack(u_s)
    AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=-1))
    return AUC, acc, f1, predictions, labels, u_s
