import time
from torch.autograd import Variable
import torch

def train_model(model, criterion, optimizer, num_epochs, dataloaders, DEVICE, name, volitile = False):
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-'*40)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
                model.eval()
                volitile = True

            running_loss = 0.0
            num = 0

            for data in dataloaders[phase]:
                input,target = data

                if torch.cuda.is_available():
                    input,target = Variable(input.to(DEVICE),volitile), Variable(target.to(DEVICE),volitile)
                else:
                    input,target= Variable(input,volitile), Variable(target,volitile)

                if phase == 'train':
                    optimizer.zero_grad()

                output = model(input, enc_mask = None)
                loss = criterion(output.float(),target.float())

                running_loss += loss * len(target)
                num += len(target)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                                    
        
            epoch_loss = running_loss / num
            print('{} Loss:  {:.4f}'.format(phase, epoch_loss))

            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
    
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # torch.save(best_model_wts, '/data/liuxiuqin/libohao/Bohao/pe/trans-crispr/param/deep_models/{}_pe_fc2'.format(name))
    model.load_state_dict(best_model_wts)
    return model

def train_deep_model(model, criterion, optimizer, num_epochs, dataloaders, DEVICE, name, volitile = False):
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-'*40)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
                model.eval()
                volitile = True

            running_loss = 0.0
            num = 0

            for data in dataloaders[phase]:
                input,target = data

                if torch.cuda.is_available():
                    input,target = Variable(input.to(DEVICE),volitile), Variable(target.to(DEVICE),volitile)
                else:
                    input,target= Variable(input,volitile), Variable(target,volitile)

                output = model(input)                
                loss = criterion(output.float(),target.float())

                running_loss += loss * len(target)
                num += len(target)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                                    
        
            epoch_loss = running_loss / num
            print('{} Loss:  {:.4f}'.format(phase, epoch_loss))

            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
    
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    torch.save(best_model_wts, '/data/liuxiuqin/libohao/Bohao/pe/trans-crispr/param/deep_models/{}_simple_pe.pkl'.format(name))
    model.load_state_dict(best_model_wts)
    return model


