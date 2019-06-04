from fastai.vision.models import resnet50
from .anchors import generate_default_anchor_maps
from numpy import int64, concatenate, arange
from torch import *
import torch
from torch import nn
from torch.nn.functional import interpolate

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
  def __init__(self, classes:int=200, topN:int=4, cat_num:int=4, im_size:tuple=(448,448)):
    super(attention_net, self).__init__()
    self.topN = topN
    self.resnet = resnet50(pretrained=True)
    self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
    self.resnet.fc = nn.Linear(512*4, classes)
    
    self.proposal_net = ProposalNet()
    self.concat_net = nn.Linear(2048 * (cat_num + 1), classes)
    self.partcls_net = nn.Linear(512*4, classes)
    _, edge_anchors, _ = generate_default_anchor_maps(input_shape=im_size)
    self.pad_size = 224
    self.edge_anchors = (edge_anchors + 224).astype(int)
    
  def forward(self, x:Tensor) -> Tensor:
    res_out, rpn_feature, feature = self.resnet(x)
    x_pad = pad(x, pad=((self.pad_side),) * 4, mode='constant', value=0)
    bs = x.size(0)
    
    rpn_score = self.proposal_net(rpn_feature.detach())    
    all_cdds = [
        concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), arange(0, len(x))
                    .reshape(-1,1)), axis=1)
        for x in rpn_score.data.cpu().numpy()]
    
    top_n_cdds = array([hard_nms(x, self.topN, 0.25) for x in all_cdds])
    
    top_n_index = top_n_cdds[:,:,-1].astype(int64)
    top_n_index = torch.from_numpy(top_n_index).cuda()
    
    top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
    part_imgs = torch.zeros([bs, self.topN, 3, 224, 224]).cuda()
    for i in range(batch):
      for j in range(self.topN):
        [y0,x0,y1,x1] = top_n_cdds[i][j, 1:5].astype(int)
        part_imgs[i:i + 1, j] = interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], 
                                            size=(224,224), mode='bilinear',
                                           align_corners=True)
        
    part_imgs = part_imgs.view(bs * self.topN, 3, 224, 224)
    _, _, part_features = self.resnet(part_imgs.detach())
    part_feature = part_features.view(batch, self.topN, -1)
    part_feature = part_feature[:, :cat_num, ...].contiguous()
    part_feature = part_feature.view(bs, -1)
    
    concat_out = torch.cat([part_feature, feature], dim=1)
    
    concat_logits = self.concat_net(concat_out)
    raw_logits = res_out
    part_logits = self.partcls_net(part_features).view(bs, self.topN, -1)
    return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]
