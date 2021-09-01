import torch

def torch_save(arr,file):
    if torch.__version__>="1.6.0":
        torch.save(arr, file, _use_new_zipfile_serialization=False)
    else:
        torch.save(arr, file)


