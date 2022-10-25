import optuna
from optuna.trial import TrialState
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import get_data
from models import CNNTransformer

DEVICE = torch.device('cuda',3) 
# if torch.cuda.is_available() else torch.device('cpu')
BATCHSIZE = 128
EPOCHS = 100

def get_dataset():
    save_dir = '/data/liuxiuqin/libohao/Bohao/pe/Data/own_split/811'
    train_loader = get_data(save_dir = save_dir, train = True, valid=False, test = False)
    valid_loader = get_data(save_dir = save_dir, train = False, valid=True, test = False)
    return train_loader, valid_loader

def define_model(trial):
    d_model = trial.suggest_int('d_model',5,30)
    d_ff = trial.suggest_int('d_ff',32,1024)
    N = trial.suggest_int('N',2,10)
    heads = trial.suggest_int('heads', 1,20)
    dropout = trial.suggest_float("dropout", 0, 0.5, step=0.05)
    activation_name = trial.suggest_categorical("activation",['GELU', 'ReLU', 'LeakyReLU', 'ReLU'])
    oc1 = trial.suggest_int('oc1', 32,256)
    oc2 = trial.suggest_int('oc2', 16,128)
    hl = trial.suggest_int('hl', 10,50)
    activation = getattr(nn, activation_name)()
    model = CNNTransformer(d_model, d_ff, N, heads, dropout, activation, oc1, oc2, hl)

    return model

def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adamax", "Adagrad", "Adadelta", "RMSprop", "Rprop", "SGD",  "ASGD"])#"LBFGS"
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the  dataset.
    train_loader, valid_loader = get_dataset()

    best_loss = float("inf")

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= len(train_loader):
                break

            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data, enc_mask = None)
            loss = F.mse_loss(output.float(), target.float())
            loss.backward()
            optimizer.step()
    
        # Validation of the model.
        model.eval()
        valid_loss = 0
        num = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= len(valid_loader):
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data, enc_mask = None)
                loss = F.mse_loss(output.float(), target.float())

                valid_loss += loss * len(target)
                num += len(target)
        total_loss = valid_loss / num
        best_loss = total_loss if total_loss < best_loss else best_loss

        trial.report(best_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            # raise optuna.exceptions.TrialPruned()
            raise optuna.TrialPruned()

    return best_loss

if __name__ == "__main__":
    study = optuna.create_study(study_name='no_seg 1.1', direction="minimize", sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.HyperbandPruner(),storage='sqlite:///db.sqlite_811')
    study.optimize(objective, n_trials=500)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))