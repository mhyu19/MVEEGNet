import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from datapipe import get_dataset
from Net import MVEEGNet 
import argparse
import random
from utils import set_random_seed, create_logger, plot_metric_history, plot_roc_curves
from evaluation import evaluate, train
import time
from plot import plot_accuracy_rejection_curve, plot_uncertainty_distribution

set_random_seed(42)


def main(args):
    logger = create_logger(args.result_dir, args.dataset, args.model, args.log)
    logger.info('Cross Validation')
    result_data = []
    best_acc_results = []

    # 为绘图准备的数据存储
    all_subjects_train_auc_history = []
    all_subjects_train_acc_history = []
    all_subjects_test_auc_history = []
    all_subjects_test_acc_history = []

    # 为ROC曲线准备的数据存储
    all_best_train_preds, all_best_train_labels = [], []
    all_best_test_preds, all_best_test_labels = [], []
    all_best_test_u, y_t, y_p = [], [], []

    for cv_n in range(args.subjects):
        best_val_acc = 0.0
        best_epoch = 0

        best_train_preds_fold, best_train_labels_fold = None, None
        best_test_preds_fold, best_test_labels_fold, best_use = None, None, None
        best_y_p, best_y_t = None, None

        current_subject_train_auc, current_subject_train_acc = [], []
        current_subject_test_auc, current_subject_test_acc = [], []

        train_dataset, test_dataset= get_dataset(args.data_dir, args.subjects, cv_n, args.dataset)

        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        target_loader = DataLoader(test_dataset, args.batch_size)
        test_loader = DataLoader(test_dataset, args.batch_size)
        if args.dataset == 'faced3' or args.dataset == 'faced9':
            model = MVEEGNet(30, 256, 32, 3, args.num_views, 32, args.classes, args.topk,args.epsilon).to(args.device)
        elif args.dataset == 'seed' or args.dataset == 'seedfold':
            model = MVEEGNet(265, 256, 32, 3, args.num_views, 62, args.classes, args.topk,args.epsilon).to(args.device)
        elif args.dataset == 'seed4'or args.dataset == 'seed4fold':
            model = MVEEGNet(64, 256, 32, 3, args.num_views, 62, args.classes, args.topk,args.epsilon).to(args.device)
        elif args.dataset == 'seed5':
            model = MVEEGNet(74, 256, 32, 3, args.num_views, 62, args.classes, args.topk,args.epsilon).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit = torch.nn.CrossEntropyLoss()

        for epoch in range(args.epochs): 
            loss = train(model, train_loader, target_loader, crit, optimizer, epoch, args)
            # loss = trainWithNoise(model, train_loader, target_loader, crit, optimizer, epoch, args)

            train_AUC, train_acc, _, train_preds, train_labels, _ = evaluate(model, train_loader, args)
            current_subject_train_auc.append(train_AUC)
            current_subject_train_acc.append(train_acc)

            test_AUC, test_acc, _, test_preds, test_labels, u_setest = evaluate(model, test_loader, args)
            y_ture = np.argmax(test_labels, axis=1)
            y_pred = np.argmax(test_preds, axis=1)
            current_subject_test_auc.append(test_AUC)
            current_subject_test_acc.append(test_acc)

            if test_acc > best_val_acc:
                # 保存模型                
                torch.save(model.state_dict(), os.path.join(args.result_dir, f'best_model_{cv_n}.pth'))
                best_val_acc = test_acc
                best_epoch = epoch + 1
                best_train_preds_fold, best_train_labels_fold = train_preds, train_labels
                best_test_preds_fold, best_test_labels_fold, best_use = test_preds, test_labels, u_setest
                best_y_p = y_pred
                best_y_t = y_ture
                best_use = best_use.reshape(-1)

            logger.info(  #, domain_loss:{domain_loss:.4f}, mmd_loss:{mmd_loss:.4f}
                f'CV{cv_n:02d}, EP{epoch + 1:03d}, Loss:{loss:.4f}')
            logger.info(
                f'Train AUC:{train_AUC*100:.2f}%, Acc:{train_acc*100:.2f}% | Test AUC:{test_AUC*100:.2f}%, Acc:{test_acc*100:.2f}% | Best Acc: {best_val_acc*100:.2f}%')

            if best_val_acc == 1:
                break

        all_subjects_train_auc_history.append(current_subject_train_auc)
        all_subjects_train_acc_history.append(current_subject_train_acc)
        all_subjects_test_auc_history.append(current_subject_test_auc)
        all_subjects_test_acc_history.append(current_subject_test_acc)

        all_best_train_preds.append(best_train_preds_fold)
        all_best_train_labels.append(best_train_labels_fold)
        all_best_test_preds.append(best_test_preds_fold)
        all_best_test_labels.append(best_test_labels_fold)
        all_best_test_u.append(best_use)
        y_t.append(best_y_t)
        y_p.append(best_y_p)

        best_acc_results.append(best_val_acc)
        result_data.append([cv_n, best_epoch, best_val_acc, test_acc])
        df = pd.DataFrame(result_data, columns=['Subject', 'Best_Epoch', 'Best_Vacc', 'Final Test_Acc'])
        df.to_csv(dfile, index=False)

    print("\n=== Final Results ===")
    logger.info(f"Mean Vacc: {np.mean(best_acc_results):.4f} ± {np.std(best_acc_results):.4f}")
    logger.info("Individual Results:")
    for subj, acc in enumerate(best_acc_results):
        logger.info(f"Subject {subj:02d}: {acc:.4f}")

    # === 绘图与数据保存部分 ===
    print("\nGenerating plots and saving data...")

    # --- 1. 处理并保存 AUC vs. Epochs 的数据 ---
    # 训练集
    train_auc_df = pd.DataFrame(all_subjects_train_auc_history).transpose()
    train_auc_df.columns = [f'Subject_{i + 1}' for i in range(args.subjects)]
    train_auc_filepath = os.path.join(args.result_dir, 'Train_AUC_vs_Epochs_Data.txt')
    train_auc_df.to_csv(train_auc_filepath, sep='\t', index_label='Epoch')
    print(f"Data for Train AUC plot saved to: {train_auc_filepath}")

    # 测试集
    test_auc_df = pd.DataFrame(all_subjects_test_auc_history).transpose()
    test_auc_df.columns = [f'Subject_{i + 1}' for i in range(args.subjects)]
    test_auc_filepath = os.path.join(args.result_dir, 'Test_AUC_vs_Epochs_Data.txt')
    test_auc_df.to_csv(test_auc_filepath, sep='\t', index_label='Epoch')
    print(f"Data for Test AUC plot saved to: {test_auc_filepath}")

    # --- 3. 绘制 AUC vs. Epochs 图 ---
    # plot_metric_history(train_auc_df, 'Train AUC vs. Epochs', 'AUC',
    #                     os.path.join(result_dir, 'Train_AUC_vs_Epochs.png'))
    plot_metric_history(test_auc_df, 'Test AUC vs. Epochs', 'AUC',
                        os.path.join(args.result_dir, 'Test_AUC_vs_Epochs.png'), args.subjects)

    # (可选) 绘制准确率曲线图
    train_acc_df = pd.DataFrame(all_subjects_train_acc_history).transpose()
    test_acc_df = pd.DataFrame(all_subjects_test_acc_history).transpose()
    train_acc_df.columns = [f'Subject_{i + 1}' for i in range(args.subjects)]
    test_acc_df.columns = [f'Subject_{i + 1}' for i in range(args.subjects)]
    plot_metric_history(train_acc_df, 'Train Accuracy vs. Epochs', 'Accuracy', os.path.join(args.result_dir, 'Train_Accuracy_vs_Epochs.png'),args.subjects)
    plot_metric_history(test_acc_df, 'Test Accuracy vs. Epochs', 'Accuracy', os.path.join(args.result_dir, 'Test_Accuracy_vs_Epochs.png'),args.subjects)

    # --- 4. 处理并保存 ROC 曲线的数据 ---
    # 训练集
    train_roc_labels = np.vstack(all_best_train_labels)
    train_roc_preds = np.vstack(all_best_train_preds)
    train_roc_labels_filepath = os.path.join(args.result_dir, 'Train_ROC_Data_Labels.txt')
    train_roc_preds_filepath = os.path.join(args.result_dir, 'Train_ROC_Data_Preds.txt')
    np.savetxt(train_roc_labels_filepath, train_roc_labels, fmt='%d')
    np.savetxt(train_roc_preds_filepath, train_roc_preds, fmt='%.8f')
    print(f"Data for Train ROC plot saved to: {train_roc_labels_filepath} and {train_roc_preds_filepath}")

    # 测试集
    test_roc_labels = np.vstack(all_best_test_labels)
    test_roc_preds = np.vstack(all_best_test_preds)
    test_roc_u = np.hstack(all_best_test_u)
    y_tr = np.hstack(y_t)
    y_pr = np.hstack(y_p)

    test_roc_labels_filepath = os.path.join(args.result_dir, 'Test_ROC_Data_Labels.txt')
    test_roc_preds_filepath = os.path.join(args.result_dir, 'Test_ROC_Data_Preds.txt')
    test_roc_u_filepath = os.path.join(args.result_dir, '不确定性.txt')
    np.savetxt(test_roc_labels_filepath, test_roc_labels, fmt='%d')
    np.savetxt(test_roc_preds_filepath, test_roc_preds, fmt='%.8f')
    np.savetxt(test_roc_u_filepath, test_roc_u, fmt='%.8f')
    plot_accuracy_rejection_curve(y_tr,  y_pr, test_roc_u, 
                                  os.path.join(args.result_dir, 'Accuracy_Rejection_Curve.png'))
    plot_uncertainty_distribution(y_tr,  y_pr, test_roc_u,
                                  os.path.join(args.result_dir, 'fig_uncertainty_dist.png'))
    print(f"Data for Test ROC plot saved to: {test_roc_labels_filepath} and {test_roc_preds_filepath}")


    # --- 6. 绘制 ROC 曲线图 ---
    # plot_roc_curves(train_roc_labels, train_roc_preds, "Train Data (from Best Epochs)",
    #                 os.path.join(result_dir, 'Train_ROC_Curve.png'))
    plot_roc_curves(test_roc_labels, test_roc_preds, "Test Data (from Best Epochs)",
                    os.path.join(args.result_dir, 'Test_ROC_Curve.png'))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # python main.py --dataset seed4 --subjects 15 --epoch 200 --classes 4 --result_dir MVEEGNet/views2 --num_views 2
    # python main.py --dataset faced --subjects 10 --epoch 200 --classes 9 --result_dir MVEEGNet/views2 --num_views 2
    # python main.py --dataset seed --subjects 15 --epoch 200 --classes 3 --result_dir MVEEGNet/views2 --num_views 2
    parser = argparse.ArgumentParser(description='MVEEGNet parameters')
    parser.add_argument('--model', type=str, default='MVEEGNet',
                        help='model name: MVEEGNet')
    parser.add_argument('--dataset', type=str, default='seed4',
                        help='the dataset used for MS-MDAER, "seed3" or "seed4", faced')
    parser.add_argument('--subjects', type=int, default='15',
                        help='Leave one cross-validation subject, use 10-fold cross-validation on the faced dataset.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='size for one batch, integer')
    parser.add_argument('--epochs', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--classes', type=int, default=3,
                        help='number of classes, integer, SEED:3, SEEDIV:4, faced3 or faced9')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--log', default='log', type=str,
                        help='output path, subdir under output_root')
    parser.add_argument('--device', type=str, default='cuda:0', help='training device')
    parser.add_argument('--data_dir', type=str, default='./dataset_EEG/', help='data path')
    parser.add_argument('--result_dir', type=str, default='./result_ASANA/', help='result save path')
    parser.add_argument('--num_views', type=int, default=5, help='number of views')
    parser.add_argument('--topk', type=int, default=10, help='topk for TGCN')
    parser.add_argument('--epsilon', type=float, default=0.5, help='epsilon for TGCN')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()
    setup_seed(args.seed)
    args.result_dir = os.path.join(args.result_dir, args.dataset)
    log_file = args.log
    os.makedirs(args.result_dir, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    dfile = os.path.join(args.result_dir, f'{args.model}_{log_file}_{time_str}.csv')  
    df = pd.DataFrame()  
    df.to_csv(dfile, index=False)  
    main(args)
