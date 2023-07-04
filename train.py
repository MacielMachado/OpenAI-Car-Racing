from data_precessing import DataHandler
from model import Model, Model2
import numpy as np
import wandb
import torch
import os

class Trainer():
    def __init__(self, X_train, y_train, X_valid, y_valid, model, optimizer, loss_func, device="mps", batch_size=64):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.batch_size = batch_size
        self.model = model.to(device)

    def wandb_init(self):
        wandb.init(
            project="OpenAI-Car-Racing",
            config={
                "loss_func": 'MSE',
                "batch_size": self.batch_size,
            }
        )

    def extract_action_MSE(self, y, y_hat):
        assert len(y) == len(y_hat)
        y_diff = y - y_hat
        y_diff_pow_2 = torch.pow(y_diff, 2)
        y_diff_sum = torch.sum(y_diff_pow_2, dim=0)/len(y)
        y_diff_sqrt = torch.pow(y_diff_sum, 0.5)
        return y_diff_sqrt

    def run(self):
        self.wandb_init()
        train_loader = self._dataloader()
        validation_loader = self._dataloader(dataset='test')
        self._training_loop(train_loader)
        wandb.finish()
        self._validation(validation_loader)
        self._save_model()

    def _save_model(self):
        os.makedirs('model_pytorch', exist_ok=True)
        torch.save(self.model.state_dict(), os.getcwd()+'/model_pytorch'+'/model_wandb.pkl')

    def _training_loop(self, train_loader):
        loss = 0
        iter = 0
        loss_bin = 0
        for index, (X, y) in enumerate(train_loader):
            self.optimizer.zero_grad()
            X = X.unsqueeze(1).float()
            y_hat = self.model(X.to(self.device))
            loss = self.loss_func(y_hat, y.to(self.device))
            action_MSE = self.extract_action_MSE(y.to(self.device), y_hat)
            loss.backward()
            self.optimizer.step()
            iter += 1
            loss_bin += loss.item()
            if index % 10 == 0:
                print(f'Batch {index} loss: {loss.item()}')

            # log metrics to wandb
            wandb.log({"loss": loss.item(),
                        "left_action_MSE": action_MSE[0],
                        "acceleration_action_MSE": action_MSE[1],
                        "right_action_MSE": action_MSE[2]})
            
            print(f'Epoch average loss: {loss_bin/iter}')

    def _validation(self, test_loader):
        loss = 0
        loss_bin = 0
        self.model.eval()
        with torch.no_grad():
            for index, (X, y) in enumerate(test_loader):
                X = X.unsqueeze(1).float()
                y_hat = self.model(X.to(self.device))
                loss = self.loss_func(y_hat, y.to(self.device))
                loss_bin += loss.item()
        print(f'Test loss {loss_bin}')
    
    def _dataloader(self, dataset='train'):
        if dataset == 'train':
            X = self.X_train
            y = self.y_train
            batch_size=self.batch_size
        elif dataset == 'test':
            X = self.X_valid
            y = self.y_valid
            batch_size=self.X_valid.shape[0]
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True)
        return data_loader


if __name__ == '__main__':
    X_train_path = os.path.join("./tutorial", 'TRAIN/states_TRAIN.npy')
    X_test_path = os.path.join("./tutorial", 'TEST/states_TEST.npy')
    y_train_path = os.path.join("./tutorial", 'TRAIN/actions_TRAIN.npy')
    y_test_path = os.path.join("./tutorial", 'TEST/actions_TEST.npy')

    X_train = DataHandler().load_data(X_train_path)
    X_test = DataHandler().load_data(X_test_path)
    y_train = DataHandler().load_data(y_train_path)
    y_test = DataHandler().load_data(y_test_path)

    X = DataHandler().append_data(X_train, X_test)
    y = DataHandler().append_data(y_train, y_test)

    frac = 0.05
    X_train, X_test = DataHandler().frac_array(frac=frac, array=X)
    y_train, y_test = DataHandler().frac_array(frac=frac, array=y)

    X_train = DataHandler().to_greyscale(X_train)
    X_test = DataHandler().to_greyscale(X_test)

    X_train = DataHandler().normalizing(X_train)
    X_test = DataHandler().normalizing(X_test)

    # Preprocess data
    model  = Model()
    lr = 1e-4
    Trainer(X_train,y_train,X_test,y_test,model,torch.optim.Adam(model.parameters(),lr=lr),torch.nn.MSELoss()).run()