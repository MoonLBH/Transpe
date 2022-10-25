# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from models import *
import torch.nn as nn
import torch.optim as optim
from utils import regression_metrics, get_data
from trainer import train_model


def get_dataset():
    save_dir = '/data/liuxiuqin/libohao/Bohao/pe/Data/own_split/811'
    train_loader = get_data(save_dir = save_dir, train = True, valid=False, test = False)
    valid_loader = get_data(save_dir = save_dir, train = False, valid=True, test = False)
    test_loader = get_data(save_dir = save_dir, train = False, valid=False, test = True)
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")
    model_name = 'cnn'

    train_loader, valid_loader, test_loader = get_dataset()
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    if model_name == 'cnn':
        model = CNNTransformer(d_model = 13, d_ff = 349, N = 2, heads = 7, dropout = 0.2, 
                            activation = nn.ReLU(), oc1 = 256, oc2 = 128, hl = 21)
        # model = CNNTransformer(d_model = 13, d_ff = 349, N = 2, heads = 7, dropout = 0.2, 
        #                     activation = nn.ReLU(), oc1 = 237, oc2 = 122, hl = 21)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model, device_ids=[0, 1, 2])   # device_ids=[0, 1, 2, 3]
        model = model.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.0002)
    
    print('{}_transformer start training'.format(model_name))
    num_epochs = 120
    trained_model = train_model(model, criterion, optimizer, num_epochs, dataloaders, 
                                DEVICE, name = model_name, volitile = False)
	    
    trained_model = trained_model.to(DEVICE)
    trained_model.eval()

    for x,y in test_loader:
        x,y = x.to(DEVICE),y.to(DEVICE)
        with torch.no_grad():
            y_pred = trained_model(x, enc_mask = None)
    y_pred = y_pred.squeeze(1).cpu().detach().numpy()
    y = y.squeeze(1).cpu().numpy()

    metric_dict = regression_metrics(y, y_pred)
    print('{}transformer:'.format(model_name),'\n',metric_dict)
        


