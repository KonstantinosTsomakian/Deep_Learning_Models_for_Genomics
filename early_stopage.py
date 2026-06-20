import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True, accuracy = True):
       
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_acc = None
        self.best_model_state = None
        self.counter = 0
        self.early_stop = False
        self.accuracy = accuracy

    

    def __call__(self, val_metric, model):
        if self.accuracy:
            if self.best_acc is None: #This loop is entered only in the first epoch
                self.best_acc = val_metric
                if self.restore_best_weights:
                    self.best_model_state = model.state_dict() # Assign the model weights to the initial model
            elif val_metric > self.best_acc + self.min_delta: # If the current accuracy is greater than the best accuracy plus the delta -> Means that the model improved
                # Improvement
                self.best_acc = val_metric #set the current acc as the best acc
                if self.restore_best_weights:
                    self.best_model_state = model.state_dict()
                self.counter = 0
            else:
                # No improvement
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.restore_best_weights:
                        print("Restoring best model weights.")
                        model.load_state_dict(self.best_model_state)
        else:
            
            if self.best_loss is None: #This loop is entered only in the first epoch
                self.best_loss = val_metric
                if self.restore_best_weights:
                    self.best_model_state = model.state_dict() # Assign the model weights to the initial model
            elif val_metric < self.best_loss - self.min_delta: # If the current accuracy is greater than the best accuracy plus the delta -> Means that the model improved
                # Improvement
                self.best_loss = val_metric #set the current acc as the best acc
                if self.restore_best_weights:
                    self.best_model_state = model.state_dict()
                self.counter = 0
            else:
                # No improvement
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.restore_best_weights:
                        print("Restoring best model weights.")
                        model.load_state_dict(self.best_model_state)
