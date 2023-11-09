import fm
import torch
from ..redevelop.data import make_data_loader

if __name__ == "__main__":
    model, alphabet = fm.downstream.build_rnafm_resnet(type="ss") 
    train_loader, val_loader, test_loader = make_data_loader()
    # for