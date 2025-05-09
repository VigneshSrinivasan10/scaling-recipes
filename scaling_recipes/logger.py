import os
from datetime import datetime

import torch

class Logger:
    def __init__(self, log_interval_epochs: int = 1, log_dir='logs', loss_dir="loss", ckpt_dir="ckpt", visuals_dir="visuals"):
        self.log_interval_epochs = log_interval_epochs
        self.log_dir = log_dir #os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.log_dir, exist_ok=True)

        self.loss_dir = os.path.join(self.log_dir, loss_dir)
        self.ckpt_dir = os.path.join(self.log_dir, ckpt_dir)
        self.visuals_dir = os.path.join(self.log_dir, visuals_dir)

        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.visuals_dir, exist_ok=True)
        
        self.train_loss_file = os.path.join(self.loss_dir, "train_loss.log")
        self.val_loss_file = os.path.join(self.loss_dir, "val_loss.log")
        self.ckpt_path = os.path.join(self.ckpt_dir, "model.pth")

    def log_train_loss(self, epoch, loss):
        with open(self.train_loss_file, 'a') as file:
            file.write(f"Epoch: {epoch}, Loss: {loss}\n")

    def log_val_loss(self, epoch, loss, accuracy):
        with open(self.val_loss_file, 'a') as file:
            file.write(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}\n")

    def save_model(self, model, optimizer):
        # Save the model and optimizer state
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, self.ckpt_path)

    def load_model(self, model):
        # Load the model and optimizer state
        checkpoint = torch.load(self.ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
                                             
    def save_visuals(self, model, dataset):
        from scaling_recipes.util import animate_flow, plot_data
        #animate_flow(model, target_file=self.flow_animation_file)
        plot_data(model, dataset, target_file=self.data_comparison_file)
    