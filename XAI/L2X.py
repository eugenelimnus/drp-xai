import omnixai
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.explainers.tabular.agnostic.L2X.l2x import L2XTabular
import argparse
import os
import sys
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
sys.setrecursionlimit(1000000)
import warnings
from sklearn.model_selection import KFold, GridSearchCV
import pandas as pd
from math import sqrt
import sklearn.preprocessing as sk
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import random
from random import randint
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data
import itertools 
from Combine import CombinedModel
from prettytable import PrettyTable


from LoadData import *
from NetVelo import *
from FuncVelov3 import *

warnings.filterwarnings("ignore")
torch.set_num_threads(64)



def main():
    train_arg_parser = argparse.ArgumentParser()
    train_arg_parser.add_argument("--drug", type=str, default='Erlotinib', help='input drug to train a model') 
    train_arg_parser.add_argument("--data_root", type=str, default='./Data/', help="path to molecular and pharmacological data")        
    train_arg_parser.add_argument("--save_logs", type=str, default='./Velodrome/logs/', help='path of folder to write log')
    train_arg_parser.add_argument("--save_models", type=str, default='./Velodrome/models/', help='folder for saving model')
    train_arg_parser.add_argument("--save_results", type=str, default='./Velodrome/results/', help='folder for saving model')
    train_arg_parser.add_argument("--hd", type=int, default=2, help='strcuture of the network')
    train_arg_parser.add_argument("--bs", type=int, default=64, help='strcuture of the network')    
    train_arg_parser.add_argument("--ldr", type=float, default=0.5, help='dropout')
    train_arg_parser.add_argument("--wd", type=float, default=0.5, help='weight decay')
    train_arg_parser.add_argument("--wd1", type=float, default=0.1, help='weight decay 1')
    train_arg_parser.add_argument("--wd2", type=float, default=0.1, help='weight decay 2')
    train_arg_parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
    train_arg_parser.add_argument("--lr1", type=float, default=0.005, help='learning rate 1')
    train_arg_parser.add_argument("--lr2", type=float, default=0.005, help='learning rate 2')    
    train_arg_parser.add_argument("--lam1", type=float, default=0.005, help='lambda 1')
    train_arg_parser.add_argument("--lam2", type=float, default=0.005, help='lambda 2')         
    train_arg_parser.add_argument("--epoch", type=int, default=5, help='number of epochs')
    train_arg_parser.add_argument("--seed", type=int, default=42, help='set the random seed')          
    train_arg_parser.add_argument('--gpu', type=int, default=0, help='set using GPU or not')

    args = train_arg_parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    
    if args.drug == "Docetaxel":
        X_tr, Y_tr, X_ts_1, Y_ts_1, X_ts_2, Y_ts_2, X_U = prep_data(args)
    else: 
        X_tr, Y_tr, X_ts_1, Y_ts_1, X_ts_2, Y_ts_2, X_ts_3, Y_ts_3, X_U, gene_names = prep_data(args)


    X_tr1 = X_tr[0]
    Y_tr1 = Y_tr[0]
    X_tr2 = X_tr[1]
    Y_tr2 = Y_tr[1]
    
    X1_train, X1_test, y1_train, y1_test = train_test_split(X_tr1, Y_tr1, test_size=0.2, random_state=args.seed, shuffle=True)    
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_tr2, Y_tr2, test_size=0.1, random_state=args.seed, shuffle=True)    
    #XU_train, XU_test = train_test_split(X_U, test_size=0.3, random_state=42, Shuffle=True)   

    X_train = np.concatenate((X1_train, X2_train, X_U), axis=0)
    X_val = np.concatenate((X1_test, X2_test), axis=0)
    y_val = np.concatenate((y1_test, y2_test), axis=0)
    
    scaler = sk.StandardScaler()
    scaler.fit(X_train)
    X1_train_N = scaler.transform(X1_train)
    X2_train_N = scaler.transform(X2_train)
    X_U_N = scaler.transform(X_U)

    train1Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X1_train_N) )#, torch.FloatTensor(y1_train))
    trainLoader_1 = torch.utils.data.DataLoader(dataset = train1Dataset, batch_size=args.bs, shuffle=True, num_workers=1)

    train2Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X2_train_N) ) #, torch.FloatTensor(y2_train))
    trainLoader_2 = torch.utils.data.DataLoader(dataset = train2Dataset, batch_size=args.bs, shuffle=True, num_workers=1)    
 
    trainUDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_U_N))
    trainULoader = torch.utils.data.DataLoader(dataset = trainUDataset, batch_size=args.bs, shuffle=True, num_workers=1)        

    model, pred1, pred2 = Network(args, X1_train_N)
    
    model.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Model.pt')))
    pred1.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Pred1.pt')))
    pred2.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Pred2.pt')))

    combine1 = CombinedModel(model, pred1)
    combine2 = CombinedModel(model, pred2)


    data1 = torch.FloatTensor(X1_train_N)
    data2 = torch.FloatTensor(X2_train_N)



    df1 = pd.DataFrame(data=X1_train_N,
                      columns=list(range(1,2067)))
    tabular_data1 = Tabular(
        df1
    )
    df2 = pd.DataFrame(data=X2_train_N,
                      columns=list(range(1,2067)))
    tabular_data2 = Tabular(
        df2
    )
    # Data preprocessing
    transformer1 = TabularTransform().fit(tabular_data1)
    transformer2 = TabularTransform().fit(tabular_data2)

    print("Initialize explainers")
    # Initialize a L2X TabularExplainer
    explainer1 = L2XTabular(
        training_data=tabular_data1, 
        predict_function=lambda z: combine1(torch.FloatTensor(transformer1.transform(z))).detach().numpy(),
        mode = "regression"
    )
    explainer2 = L2XTabular(
        training_data=tabular_data2, 
        predict_function=lambda z: combine2(torch.FloatTensor(transformer2.transform(z))).detach().numpy(),
        mode = "regression"
    )

    print("Generate explanation example 1")
    # Generate explanations
    test_instance1 = tabular_data1[:1]
    test_instance2 = tabular_data2[:1]
    local_explanations1 = explainer1.explain(test_instance1)
    l1dict = local_explanations1.get_explanations(index=0)
    features = l1dict['features'][:10]
    print("Average magnitude of Importance score:")
    print(sum(list(map(lambda x:abs(x),l1dict['scores']))) / len(l1dict['scores']))
    l1feature_names = []
    for i in features:  
        l1feature_names.append(gene_names[i-1])
    scores = l1dict['scores'][:10]
    res = zip(features, l1feature_names, scores)
    table1 = PrettyTable()
    table1.field_names = ['Feature no.', 'Gene name', 'Importance score']
    for i in res:
        table1.add_row(list(i))
    print(table1)

    print("Generate explanation example 2")
    local_explanations2 = explainer2.explain(test_instance2)
    l2dict = local_explanations2.get_explanations(index=0)
    features2 = l2dict['features'][:10]
    print("Average Importance score:")
    print(sum(l2dict['scores']) / len(l2dict['scores']))
    print("Top 10 features and their corresponding gene name:")
    print(features2)
    l2feature_names = []
    for i in features2:
        l2feature_names.append(gene_names[i-1])
    print(l2feature_names)
    scores2 = l2dict['scores'][:10]
    print("Importance scores:")
    print(scores2)


main()


