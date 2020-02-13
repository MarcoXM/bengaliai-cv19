import torch
import torch.nn as nn
import sklearn.metrics
import numpy as np 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion_1 = nn.CrossEntropyLoss().to(device, torch.float32)
criterion_2 = nn.CrossEntropyLoss().to(device, torch.float32)
criterion_3 = nn.CrossEntropyLoss().to(device, torch.float32)

def calc_advloss_D(output,target, c1= criterion_1,c2= criterion_2,c3= criterion_3):
    
    l1 = c1(output[0], target[:,0])
    l2 = c2(output[1], target[:,1])
    l3 = c3(output[2], target[:,2])
    loss = l1 + l2 + l3
    return loss

def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()
    # pred_y = [p.cpu().numpy() for p in pred_y]

    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    # print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '
    #       f'total {final_score}, y {y.shape}')
    return final_score