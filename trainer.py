import torch
from sklearn.metrics import f1_score
from torchmetrics import F1Score
from tqdm.autonotebook import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os

# Model was trained locally on apple silicon since we only train 20-50 epochs
# change if you do not want to train on apple silicon

mps_device = torch.device("mps")

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 scheduler=None,               # Scheduler for the LR
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._scheduler = scheduler
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        
        # set yor directory
        new_working_directory = "/Users/sebh/Desktop/GithubProjectDL/SolarPanel_defect_detection"
        os.chdir(new_working_directory)
        
        # Tensorboard logging and checkpoint save
        log_dir = f"runs/train_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._writer = SummaryWriter(log_dir=log_dir)
    
        # change to cude if you do not want to train on apple silicon
        if cuda:
            # self._model = model.cuda()
            # self._crit = crit.cuda()
            self._model = model.to(mps_device)
            self._crit = crit.to(mps_device)
            
            #checkpoints
  
    def save_checkpoint(self, epoch):
        checkpoint_file = f"{self._writer.log_dir}/checkpoint_{epoch}.ckp"
        torch.save({'state_dict': self._model.state_dict()}, checkpoint_file)

    def restore_checkpoint(self, epoch_n):
        # specify the name (path) of the checkpoint you would like to load
        ckp = torch.load(f'runs/train_best2_load/checkpoint_{epoch_n}.ckp', map_location=torch.device('cpu'))
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        torch.onnx.export(m,             # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})


    def train_step(self, x, y):
        self._optim.zero_grad()

        self._model.train()
        self._train_dl.mode = "train"

        out = self._model(x)
        loss = self._crit(out, y)
        loss.backward()
        self._optim.step()

        return loss 

        
    def val_test_step(self, x, y):
        self._model.eval()

        with torch.no_grad():
            preds = self._model(x)
            loss = self._crit(preds, y)

        return loss, preds

        
    def train_epoch(self):
        running_loss = 0.0
        n_batches = 0
        self._model.train()
        self._train_dl.mode = "train"

        for inputs, labels in self._train_dl:
            
            #inputs, labels = (inputs.cuda(), labels.cuda()) if self._cuda else (inputs, labels)
            inputs, labels = (inputs.to(mps_device), labels.to(mps_device)) if self._cuda else (inputs, labels)
            
            # Transfer the batch to the GPU if it's available
            #inputs = inputs.to(self._cuda) #specify mps device later
            #labels = labels.to(self._cuda)

            loss = self.train_step(inputs, labels)
            running_loss += loss

            n_batches += 1


        for scheduler in self._scheduler:
            scheduler.step()

        #self._scheduler.step()
        avg_loss = running_loss / n_batches

        self._writer.add_scalar('Loss/train', avg_loss, self.epoch_counter)
        
        return avg_loss
    
    def val_test(self):
        pred_list = []
        label_list = []
        running_loss = 0

        self._model.eval()
        self._val_test_dl.mode = "val"

        with torch.no_grad():
            for inputs, labels in self._val_test_dl:
                #inputs, labels = (inputs.cuda(), labels.cuda()) if self._cuda else (inputs, labels)
                inputs, labels = (inputs.to(mps_device), labels.to(mps_device)) if self._cuda else (inputs, labels)
                
                # transfer the batch to the gpu if given
                label_list.append(labels.cpu().numpy())
    
                loss, preds = self.val_test_step(inputs, labels)
                pred_list.append(np.around(preds.cpu().numpy()))

                running_loss += loss

        return self.loss_metrics(running_loss, np.array(pred_list), np.array(label_list))


    def loss_metrics(self, sum_loss, out_pred, out_true):
        # calculate the average loss and average metrics of your choice. 
        # You might want to calculate these metrics in designated functions

        # calculate average loss
        avg_loss = sum_loss / len(self._val_test_dl)

        #calculate the F1 score here
        f1_crack = f1_score(out_pred[:, :, 0].flatten(), out_true[:, :, 0].flatten(), average='binary') # indexing might be wrong
        f1_inactive = f1_score(out_pred[:, :, 1].flatten(), out_true[:, :, 1].flatten(), average='binary')
        f1_mean = (f1_crack + f1_inactive) / 2

        # log everything to tensorboard
        self._writer.add_scalar('Loss/val', avg_loss, self.epoch_counter)  # log the validation loss to TensorBoard
        self._writer.add_scalar('F1Score/Average', f1_mean, self.epoch_counter)  # log the average F1 score to TensorBoard
        self._writer.add_scalar('F1Score/Crack', f1_crack, self.epoch_counter)  # log the F1 score for "crack" to TensorBoard
        self._writer.add_scalar('F1Score/Inactive', f1_inactive, self.epoch_counter)  # log the F1 score for "inactive" to TensorBoard
        self._writer.add_scalar('LR', self._optim.param_groups[0]['lr'], self.epoch_counter)

        return avg_loss, f1_mean, f1_crack, f1_inactive

    def fit(self, epochs=-1):
        
        # Commented out to prevent restoring from a checkpoint
        # specify epoch number from which you want to restore the model's state.
        # You can only restore from epochs that you saved!
        #self.restore_checkpoint(7)
        
        assert self._early_stopping_patience > 0 or epochs > 0
        self.epoch_counter = 0
        train_loss = []
        val_loss = []
        f1_val_avg = []
        f1_val_crack = []
        f1_val_inactive = []
        
        # Initialize best_loss to inf and patience_counter to 0
        best_loss = np.inf
        patience_counter = 0

        while True:
            if self.epoch_counter == epochs:
                break

            train_loss.append(self.train_epoch())
            avg_loss, f1_mean, f1_crack, f1_inactive = self.val_test()

            #self._scheduler.step(avg_loss)
            val_loss.append(avg_loss)
            f1_val_avg.append(f1_mean)
            f1_val_crack.append(f1_crack)
            f1_val_inactive.append(f1_inactive)

            print ('--------------', self.epoch_counter, '--------------')
            print('val_loss:', val_loss[-1].item())
            print('f1_mean', f1_val_avg[-1])
            print('f1_crack', f1_val_crack[-1])
            print('f1_inactive', f1_val_inactive[-1])
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save the best model so far
                self.save_checkpoint(self.epoch_counter)
            else:
                patience_counter += 1
                if patience_counter >= self._early_stopping_patience:
                    print("Early stopping due to no improvement after", self._early_stopping_patience, "epochs.")
                    break

            self.epoch_counter += 1

        self.save_checkpoint(self.epoch_counter)

        self._writer.close()

        return train_loss[-1], val_loss[-1]