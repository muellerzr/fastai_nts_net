from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from fastai.torch_core import *
from fastai.basic_train import *
from fastai.basic_data import *
from fastai.layers import *
from fastai.callback import *
from fastai.layers import *
from fastai.callbacks.hooks import *

drivedownloader = try_import('google_drive_downloader')
if not drivedownloader:
    raise Exception('Error: `googledrivedownloader` is needed. `pip install googledrivedownloader`')
from google_drive_downloader import GoogleDriveDownloader as gdd
                    
from .resnet import *
from .anchors import generate_default_anchor_maps, hard_nms


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class attention_net(nn.Module):
    def __init__(self, topN=4, classes:int=200, cat_num:int=4):
        super(attention_net, self).__init__()
        self.cat_num=cat_num
        self.pretrained_model = resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.pretrained_model.fc = nn.Linear(512 * 4, 200)
        self.pretrained_model.fc = nn.Linear(512 * 4, classes)
        self.proposal_net = ProposalNet()
        self.topN = topN
        # self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), 200)
        self.concat_net = nn.Linear(2048 * (self.cat_num + 1), classes)
        # self.partcls_net = nn.Linear(512 * 4, 200)
        self.partcls_net = nn.Linear(512 * 4, classes)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int64) # when running code, here went a error, change np.int to np.int64,parameter index of torch.gather() appoint longtensortype when change num_worker to 4,then it runs on windows or linux correctly
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :self.cat_num, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        # part_logits have the shape: B*N*200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]

def _nts_body_cut(m:nn.Module):
    return [[*list(m.pretrained_model.children())[:8]]]

def _nts_cut(m:nn.Module)->List[nn.Module]:
    groups = [[*list(m.pretrained_model.children())[:8]]]
    groups += [[*list(m.pretrained_model.children())[8:], m.children()[1:]]]
    return groups

def get_body(topN:int=4, cat_num:int=4, pretrained:bool=True, data:DataBunch):
    if pretrained:
        path = data.path
        net = attention_net(topN,200,cat_num)
        gdd.download_file_from_google_drive(file_id='1Nbc9HMt4YPd2Wjri6BCCiTygUhTaPdxA', dest_path=Path(path/'Pretrained-Weights.pth'))
        net.load_state_dict(torch.load('Pretrained_Weights.pth'))['net_state_dict']                                            
    else:
        net = attention_net(6, 200, 4)
    body = create_body(attention_net, cut=_nts_body_cut)
    return body

def get_head(nc:int=200):
    
    h1 = [*list(net.pretrained_model.children())[8:]]
    h1[1] = nn.Linear(2048, nc)
    cn = nn.Linear(10240, nc)
    prt = nn.Linear(2048, nc)
    
    head = nn.Sequential(h1, ProposalNet(), cn, prt)
    return head

def nts_learner(data:DataBunch, topN:int=4, cat_num:int=4, pretrained:bool=True, **kwargs:Any)->Learner:
    'Build a convnet style learner for NTS-Net'
    body = get_body(topN, cat_num, pretrained, data)
    head = get_head(data.c)
    model = nn.Sequential(body, head)
    learn = Learner(data, model, **kwargs)
    learn.split(_nts_cut)
    if pretrained: learn.freeze()
    if init: apply_init(model[1], init)
    return learn



