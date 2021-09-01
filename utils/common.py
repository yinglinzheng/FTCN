#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import torch
from torch.autograd import Variable
import errno
import torch.distributed as dist
import math
from functools import reduce
def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            print(os.path.join(path, version))
            os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def skipShardSplit(aList, drop_last=False, num_replicas=None, rank=None):
    if not isinstance(aList, list) and not isinstance(aList, tuple):
        aList = List

    if num_replicas is None:
        num_replicas = dist.get_world_size() if dist.is_initialized() else 1
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0

    num_replicas = num_replicas
    rank = rank
    drop_last = drop_last

    if drop_last:
        aList = aList[0: (len(aList) // num_replicas) * num_replicas]

    # subsample
    aList = aList[rank::num_replicas]

    return aList

def mixb2a(a,b):
    if len(b) > len(a):
        a,b = b,a
    if len(b) == 0:
        return a
    chunk_num = (len(b))
    a_chunk = splitIntoChunk(a, chunk_num)
    b_chunk = list(map(lambda x:[x],b))
    return reduce(lambda x, y: x+y, [_a+_b for _a,_b in zip(a_chunk, b_chunk)])

def splitIntoChunk(aList, chunk_num):
    return [aList[math.ceil(k * (len(aList) / chunk_num)):math.ceil((k + 1) * (len(aList) / chunk_num)):] for k in range(chunk_num)]