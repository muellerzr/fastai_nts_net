from fastai import *
from fastai.core import *
from fastai.vision import *
import fastai.metrics as metrics

from torch import sum
from torch.nn.functional import cross_entropy as CRE
from torch.nn.functional import relu
from .prediction import *


def total_loss(out, label):
    bs = learn.data.batch_size
    
    raw_logits, concat_logits, part_logits, _, top_n_prob = out
    
    lbl = label.unsqueeze(1).repeat(1, 6).view(-1)
    lgt = part_logits.view(bs * 6, -1)
    
    part_loss = list_loss(lgt, lbl).view(bs,6)
    raw_loss = CRE(raw_logits, label)
    concat_loss = CRE(concat_logits, label)
    rank_loss = ranking_loss(top_n_prob, part_loss, 6)
    partcls_loss = CRE(lgt, lbl)
    
    total_loss = rank_loss + raw_loss + concat_loss + partcls_loss
    return total_loss.squeeze(0)
    
def ranking_loss(score, targets, proposal_num):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = sum(relu(loss_p))
        loss += loss_p
        
    return loss / batch_size
    
def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)

    
def metric(out, label):
     
    return metrics.accuracy(get_pred(out), label)
    
