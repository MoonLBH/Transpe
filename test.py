from utils import regression_metrics, get_test_dataset
import torch
from models import *
import pandas as pd

def get_dataset():
    test_loader_dict = get_test_dataset(['Endo', 'HCT116', 'MDA'])
    return test_loader_dict

if __name__ == "__main__":
    DEVICE = torch.device('cuda', 0)
    model = CNNTransformer(d_model = 26, d_ff = 60, N = 3, heads = 14, dropout = 0.2, 
                           activation = nn.ReLU(), oc1 = 104, oc2 = 70, hl = 23)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    param_path = '/data/liuxiuqin/libohao/Bohao/pe/trans-crispr/param/transpe_param.pkl'
    param = torch.load(param_path)
    model.load_state_dict(param)

    model.eval()
    test_loader_dict = get_dataset()
    # for x,y in test_loader:
    #     x,y = x.to(DEVICE),y.to(DEVICE)
    #     with torch.no_grad():
    #         y_pred = model(x, enc_mask = None)
    # y_pred = y_pred.squeeze(1).cpu().detach().numpy()
    # y = y.squeeze(1).cpu().numpy()

    # metric_dict = regression_metrics(y, y_pred)
    # print('{}transformer:'.format(model_name),'\n',metric_dict)


#---------------------------------------------------------------------
# 批量test
    record_list = []
    for set_name in test_loader_dict:
        for x,y in test_loader_dict[set_name]:
            x,y = x.to(DEVICE),y.to(DEVICE)
            with torch.no_grad():
                y_pred = model(x, enc_mask = None)
        y_pred = y_pred.squeeze(1).cpu().detach().numpy()
        y = y.squeeze(1).cpu().numpy()
        metric_dict = regression_metrics(y, y_pred)
        print('{}transformer is tested in {}:'.format(model_name,set_name),'\n',metric_dict,'\n\n')
        record_dict = {'datasets name':set_name}
        record_dict.update(metric_dict)
        record_list.append(record_dict)
        
    df = pd.DataFrame(record_list)
    file_path = pd.ExcelWriter('/data/liuxiuqin/libohao/Bohao/pe/trans-crispr/general_test/dp.xlsx')
    df.to_excel(file_path,encoding = 'utf-8',index = False)
    file_path.save()
