from fastai.core import *
from fastai.vision import *
import fastai.metrics as metrics

from torch import sum
from torch.nn import CrossEntropyLoss as CRE
from torch.nn.functional import relu

from .loss_functions import *


def get_pred(out):
    _, concat_logits, _, _, _ = out
    return concat_logits.argmax()

def predict(learn:Learner, item:ItemBase):
    batch = learn.data.one_item(item)
    res = learn.pred_batch(batch=batch)
    raw_pred, x = grab_idx(res, 0),batch[0]
    norm = getattr(learn.data, 'norm', False)
    if norm:
        x = learn.data.denorm(x)
        if norm.keywords.get('do_y', False): pred = learn.data.denorm(pred)
    ds = learn.data.single_ds
    pred = get_pred(raw_pred)
    out = ds.y.reconstruct(pred, ds.x.reconstruct(img.data)) if has_arg(ds.y.reconstruct, 'x') else ds.y.reconstruct(pred)
    x = ds.x.reconstruct(grab_idx(x, 0))
    y = ds.y.reconstruct(pred, x)  if has_arg(ds.y.reconstruct, 'x') else ds.y.reconstruct(pred)
    return (x, y, pred, raw_pred)
        
