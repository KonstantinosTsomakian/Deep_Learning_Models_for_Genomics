import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from tqdm import tqdm
from early_stopage import EarlyStopping
import numpy as np
import pandas as pd

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import os 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This trainer is used to train the model that takes as input a multichannel tensor of histone modifications.
# The input is of shape [batch,channels,width] and the targets are of shape [batch,target_width]. Both inputs and targets should be of dtype .folat(s)
class histonCNN_signal_regression_trainer:
    def __init__(self, model, train_loader, test_loader, epochs=20, lr=0.0001, weight_decay = 0, early_stopping = False):
        #Define the parameters

        self.model = model
        self.train_loader = train_loader 
        self.test_loader = test_loader
        self.epochs = epochs
        self.lossfun = nn.MSELoss() #Cross entropy loss for classification tasks
        self.optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=weight_decay) # Define the optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to GPU if available
        self.early_stopping = early_stopping
        if self.early_stopping:
             self.early_stoper = EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)


	# The main function used to train the model.
    def train(self):
    
        losses = [] #Stores the losses of all epochs.
        test_losses = []
        accuracies = []
        batch_acc = [] #Stores the losses of all the batches.
        epoch_mean_batch_acc = []

        # Loop over epochs
        for epoch in tqdm(range(self.epochs)):
            

            running_loss = 0.0 # The loss of the entire epoch.
            self.model.train()  # Set model to training mode

            # Loop over minibatches in the train loader
            for regions, labels in self.train_loader:
                
                regions, labels = regions.to(self.device), labels.to(self.device) #Move data to device


                raw_logits, embedding = self.model(regions)

                ## Important: The loss in binary classification is compouted from the raw logit and the tru label. Not the predicted label and the true label

                self.optimizer.zero_grad()  # Clear gradients from previous loops
                loss = self.lossfun(raw_logits, labels) #Compute losses.
                loss.backward() # Backpropagate to compute gradients of all the model parameters
                self.optimizer.step() #Update parameters

                running_loss += loss.item()
                batch_accuracy = self.batch_regress(X=regions, y=labels)
                batch_acc.append(batch_accuracy)

            # Append the sum of losses to the list that stores the loss from each epoch
            losses.append(running_loss)
            test_epoch_logits, test_target, epoch_acc = self.regress()

            epoch_test_loss = self.lossfun(test_epoch_logits, torch.tensor(test_target).to(self.device))

            mean_epoch_accuracy_from_batches = np.mean(batch_acc).item()
            if self.early_stopping:
                 self.early_stoper(epoch_acc, self.model)
                 if self.early_stoper.early_stop:
                      break  

            epoch_mean_batch_acc.append(mean_epoch_accuracy_from_batches)
            test_losses.append(epoch_test_loss.item())
            accuracies.append(epoch_acc)
        return(accuracies, epoch_mean_batch_acc, losses, test_losses)

        # return(losses)

            

    def regress(self):

        """ Evaluate the model on the test set """
        self.model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            region, target = next(iter(self.test_loader))
            # r2_score= r2_score.to(self.device)
# 
            # Move data to device
            region, target = region.to(self.device), target.to(self.device)

            #Compute the raw logits
            raw_logit, embed = self.model(region)
            # print(raw_logit.shape, label.shape)

            # prediction = torch.relu(raw_logit)
            prediction = raw_logit

            # Because r2_score sklearn function expects arrays in the cpu i have to move themn to the cpu
            target, prediction = target.cpu().numpy(), prediction.cpu().numpy()
            r2 = r2_score(target, prediction) 
        
        accuracy = r2
        # print(accuracy)
        # print(f"Test Accuracy: {accuracy:.2f}%")
        self.model.train()  # Set model to train mode
        return(raw_logit, target, accuracy)
    

    def batch_regress(self, X ,y):

        """ Evaluate the model on the test set """
        self.model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            
                region, target = X, y
                # Move data to device
                region, target = region.to(self.device), target.to(self.device)
                

                #Compute the raw logits
                raw_logit, embed = self.model(region)
                #

                # prediction = torch.relu(raw_logit)
                prediction = raw_logit

                # Because r2_score sklearn function expects arrays in the cpu i have to move themn to the cpu
                target, prediction = target.cpu().numpy(), prediction.cpu().numpy()
                r2 = r2_score(target, prediction) 
        
        
        epoch_accuracy = r2
        
        # print(accuracy)
        # print(f"Test Accuracy: {accuracy:.2f}%")
        self.model.train()  # Set model to train mode
        return(epoch_accuracy)
    

    def save_model(self, path="model.pth"):
        """ Save the trained model """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="model.pth"):
        """ Load a trained model """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

# trainer_for_cnn_without_sequence_module
class histonCNN_trainer_without_sequence_module:
    def __init__(self, model, train_loader, test_loader,
                    input_channels,task = 'classify',
                      epochs=5, lr=0.001, weight_decay = 0,early_stopping = True):
                      
        #Define the parameters
        # -> Learning rate is defined here
        # -> The loss function is also defined here
        # -> The criterion is also defined here

        self.model = model
        self.train_loader = train_loader 
        self.test_loader = test_loader
        self.input_channels = input_channels
        self.epochs = epochs
        self.task = task
        self.optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=weight_decay) # Define the optimizer
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to GPU if available
        self.early_stopping = early_stopping
        if self.early_stopping:
             self.early_stoper = EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)        
        if self.task == 'regress':
            self.lossfun = nn.MSELoss()
        elif self.task == 'classify':
            self.lossfun = nn.BCEWithLogitsLoss() #Cross entropy loss for classification tasks


    def train(self):
        """ Train the model over multiple epochs """
        losses = []
        test_accuracies = []
        epoch_mean_batch_acc = []

        # Loop over epochs
        for epoch in tqdm(range(self.epochs)):
            
            
            running_loss = 0.0
            self.model.train()  # Set model to training mode

            # Loop over minibatches in the train loader
            for hm in self.train_loader:
                regions, labels = hm

                regions, labels = regions.to(self.device), labels.to(self.device) #Move data to device


                raw_logits, embeddings = self.model(regions)



                ## Important: The loss in binary classification is compouted from the raw logit and the tru label. Not the predicted label and the true label

                self.optimizer.zero_grad()  # Clear gradients from previous loops
                loss = (self.lossfun(raw_logits, labels))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # batch_acc.append(batch_accuracy)
                

            # Append the sum of losses to the list that stores the loss from each epoch
            losses.append(running_loss / self.train_loader.batch_size)

            if self.task == 'regress':
                epoch_acc_for_test,raw_logit, prediction, target = self.regress()
                epoch_acc_for_train = self.batch_regress(X=regions, y=labels)
            elif self.task == 'classify':
                # print('classification')
                epoch_acc_for_test,raw_logit, prediction, target = self.evaluate()
                epoch_acc_for_train = self.batch_evaluate(X=regions, y=labels)

        
            test_accuracies.append(epoch_acc_for_test)
            epoch_mean_batch_acc.append(epoch_acc_for_train)
            self.scheduler.step()
            if self.early_stopping:
                 self.early_stoper(epoch_acc_for_test, self.model)
                 if self.early_stoper.early_stop:
                      break  
            
        return(test_accuracies, epoch_mean_batch_acc, losses, raw_logit, prediction, target)

        # return(losses)

            

    def evaluate(self):

        """ Evaluate the model on the test set """
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            region, label = next(iter(self.test_loader))

            # Move data to device
            region, label = region.to(self.device), label.to(self.device)

            #Compute the raw logits
            raw_logit, embedding = self.model(region)
            # print(raw_logit)
            # print(raw_logit.shape, label.shape)

            prediction = (torch.sigmoid(raw_logit) > 0.5).long()
            # print(prediction)
            

            
            correct += (prediction == label).sum().item()
            # print(prediction, label, correct)
            total += label.size(0)
                
        
        accuracy = 100 * correct / total
        # print(f"Test Accuracy: {accuracy:.2f}%")
        self.model.train()  # Set model to train mode
        return(accuracy,raw_logit, prediction, label)
    
    def regress(self):

        """ Evaluate the model on the test set """
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            region, target = next(iter(self.test_loader))
            # r2_score= r2_score.to(self.device)

            # Move data to device
            region, target = region.to(self.device), target.to(self.device)

            #Compute the raw logits
            raw_logit, embedding = self.model(region)
            # print(raw_logit.shape, label.shape)

            # prediction = torch.relu(raw_logit)
            prediction = raw_logit
            # prediction = F.leaky_relu(raw_logit, negative_slope=0.1)

            # Because r2_score sklearn function expects arrays in the cpu i have to move themn to the cpu
            target, prediction = target.cpu().numpy(), prediction.cpu().numpy()
            r2 = r2_score(target, prediction) 
        
        accuracy = r2
        # print(accuracy)
        # print(f"Test Accuracy: {accuracy:.2f}%")
        self.model.train()  # Set model to train mode
        return(accuracy,raw_logit, prediction, target)
    

    def batch_regress(self, X ,y):

        """ Evaluate the model on the test set """
        self.model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            
                region, target = X, y
                # Move data to device
                region, target = region.to(self.device), target.to(self.device)

                #Compute the raw logits
                raw_logit, embedding = self.model(region)
                #

                prediction = torch.relu(raw_logit)

                # Because r2_score sklearn function expects arrays in the cpu i have to move themn to the cpu
                target, prediction = target.cpu().numpy(), prediction.cpu().numpy()
                r2 = r2_score(target, prediction) 
        
        
        epoch_accuracy = r2
        
        # print(accuracy)
        # print(f"Test Accuracy: {accuracy:.2f}%")
        self.model.train()  # Set model to train mode
        return(epoch_accuracy)
    
    def batch_evaluate(self, X ,y):

        
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            
                region, label = X, y
                # Move data to device
                region, label = region.to(self.device), label.to(self.device)

                #Compute the raw logits
                raw_logit, embedding = self.model(region)
                # print(raw_logit.shape, label.shape)

                prediction = (torch.sigmoid(raw_logit) > 0.5).long()
                

                
                correct += (prediction == label).sum().item()
                # print(prediction, label, correct)
                
                total += label.size(0)
                
        
        b_accuracy = 100 * correct / total
        # print(f"Test Accuracy: {accuracy:.2f}%")
        self.model.train()  # Set model to train mode
        return(b_accuracy)



class train_bert_for_mlm:
	def __init__(self, bert_mlm_model, train_loader, test_loader, dir_to_save, epochs = 10, lr = 1e-5, weight_decay = 1e-8, save_every=10):
		self.bert_mlm_model = bert_mlm_model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.epochs = epochs
		self.lr = lr
		self.weight_decay = weight_decay
		self.dir_to_save = dir_to_save
		self.early_stoper = EarlyStopping(patience=10, min_delta=0.01, restore_best_weights=True, accuracy=True)
		self.save_every = save_every
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
		self.optimizer = torch.optim.AdamW(bert_mlm_model.parameters(), self.lr, weight_decay=self.weight_decay)

          
		self.bert_mlm_model.to(device)
		
	def train(self):
		train_losses = []
		test_losses = []
		test_accuracies = []

		# train_loader, test_loader = self.train_loader.to(device), self.test_loader.to(device)
		for epoch in tqdm(range(self.epochs)):
			epoch_number = epoch + 1
			running_loss = 0
			self.bert_mlm_model.train()
			for batch_input, batch_labels in tqdm(self.train_loader):
				batch_input, batch_labels = batch_input.to(device), batch_labels.to(device)
				model_output = self.bert_mlm_model(input_ids=batch_input, labels = batch_labels )

				self.optimizer.zero_grad()

				batch_loss = model_output.loss
				batch_loss.backward()
				self.optimizer.step()

				

				
				
				
				
				
				running_loss += batch_loss
			epoch_loss = running_loss.detach() / len(self.train_loader)

			epoch_test_acc, epoch_test_loss = self.eval()

			train_losses.append(epoch_loss.detach().cpu().item())
			test_losses.append(epoch_test_loss.detach().cpu().item())
			test_accuracies.append(epoch_test_acc)
			model_weights = self.bert_mlm_model.state_dict()
			self.early_stoper(epoch_test_acc, self.bert_mlm_model)
			if self.early_stoper.early_stop:
				model_weights = self.early_stoper.best_model_state
				break  

			if (epoch_number % self.save_every) == 0:
				if not os.path.exists(self.dir_to_save):
					os.makedirs(f"{self.dir_to_save}/models", exist_ok=True)
					os.makedirs(f"{self.dir_to_save}/results", exist_ok=True)

				report_df = pd.DataFrame({
					'train_loss' : train_losses,
					'test_loss' : test_losses,
					'test_accuracy' : test_accuracies
				})

				report_df.to_csv(f'{self.dir_to_save}/results/report_epoch{epoch_number}.tsv', header=True, index=False, sep = '\t')
				torch.save(self.bert_mlm_model.state_dict(), f'{self.dir_to_save}/models/report_epoch{epoch_number}.bertmodel')




		return(train_losses, test_losses, test_accuracies, model_weights)

	def eval(self):
		self.bert_mlm_model.eval()
		total_loss = 0.0
		total_correct = 0
		total_tokens = 0

		with torch.no_grad():
			for test_batch_input, test_batch_labels in self.test_loader:
				test_batch_input = test_batch_input.to(self.device)
				test_batch_labels = test_batch_labels.to(self.device)

				outputs = self.bert_mlm_model(input_ids=test_batch_input, labels=test_batch_labels)
				total_loss += outputs.loss.item()

				predictions = torch.argmax(outputs.logits, dim=-1)
				mask = test_batch_labels != -100  # Ignore padding/masked tokens
				correct = (predictions == test_batch_labels) & mask

				total_correct += correct.sum().item()
				total_tokens += mask.sum().item()

		avg_loss = total_loss / len(self.test_loader)
		accuracy = total_correct / total_tokens

		self.bert_mlm_model.train()  # Return to training mode
		return accuracy, torch.tensor(avg_loss, device=self.device)



################## This is a simple trainer. It takes a model that outputs only a single logit ####################
# It can be used both for classsification and regression
#####################################################################
class simple_trainer:
	def __init__(self, model, train_loader, test_loader,
					epochs=5, lr=0.001, weight_decay = 0, stopper_delta = 0.01, mode = 'classify', return_probs = False, use_early_stopping = True):
  

		self.model = model
		self.train_loader = train_loader 
		self.test_loader = test_loader
		self.epochs = epochs
		self.optimizer = torch.optim.AdamW(model.parameters(),lr, weight_decay=weight_decay) # Define the optimizer
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)  # Move model to GPU if available
		self.stopper_delta = stopper_delta
		self.early_stoper = EarlyStopping(patience=5, min_delta=self.stopper_delta, restore_best_weights=True,accuracy=True)
		self.mode = mode
		self.use_early_stopping = use_early_stopping
		self.return_probs = return_probs

		if self.mode == 'classify':
			self.lossfun = nn.BCEWithLogitsLoss() #Binary cross entropy loss for classification tasks
		elif self.mode == 'regress':
			self.lossfun = nn.MSELoss() 
			print('regression')
		


	def train(self):
		""" Train the model over multiple epochs """
		losses = []
		test_accuracies = []
		epoch_mean_batch_acc = []
		epoch_mean_test_losses = []

		# Loop over epochs
		for epoch in tqdm(range(self.epochs)):


			running_loss = 0.0
			self.model.train()  # Set model to training mode

			# Loop over minibatches in the train loader
			for hm in self.train_loader:
				regions, labels = hm

				regions, labels = regions.to(self.device), labels.to(self.device).to(torch.float) #Move data to device
				


				raw_logits = self.model(regions)



				## Important: The loss in binary classification is compouted from the raw logit and the tru label. Not the predicted label and the true label

				self.optimizer.zero_grad()  # Clear gradients from previous loops
				loss = (self.lossfun(raw_logits, labels))
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()
				
				 
				
				# batch_accuracy = self.batch_evaluate(X=regions, y=labels, sequence=sequence)
				# batch_acc.append(batch_accuracy)
				




			# print('classification')
			if self.mode == 'classify':
				
				epoch_acc_for_test, test_loss, total_predictions, total_test_labels, probs = self.evaluate()        
			elif self.mode == 'regress':
				epoch_acc_for_test, test_loss, total_predictions, total_test_labels = self.regress()
			
			# print(epoch_acc_for_test)

			model_weights = self.model.state_dict()
			if self.use_early_stopping:
				self.early_stoper(epoch_acc_for_test, self.model)
				if self.early_stoper.early_stop:
					model_weights = self.early_stoper.best_model_state
					break
                  
			# Append the sum of losses to the list that stores the loss from each epoch
			losses.append(running_loss / self.train_loader.batch_size)

			test_accuracies.append(epoch_acc_for_test)
			epoch_mean_test_losses.append(test_loss)
			
		if self.return_probs:
			return(test_accuracies, epoch_mean_batch_acc, losses, epoch_mean_test_losses, self.model, total_predictions, total_test_labels, probs)
		else:

			return(test_accuracies, epoch_mean_batch_acc, losses, epoch_mean_test_losses, self.model, total_predictions, total_test_labels)

			# return(losses)

				
	def evaluate(self):

		""" Evaluate the model on the test set """
		self.model.eval()  # Set model to evaluation mode

		correct = 0
		total = 0
		sum_test_loss = 0
		total_predictions = []
		total_test_labels = []
		with torch.no_grad():
			for region, label in self.test_loader:
				
				# Move data to device
				region, label = region.to(self.device), label.to(self.device).to(torch.float)

				#Compute the raw logits
				raw_logit = self.model(region)
				test_loss = self.lossfun(raw_logit, label)
				sum_test_loss += test_loss.detach().item()
				# print(test_loss)
				# print(raw_logit)
				# print(raw_logit.shape, label.shape)
                        
                

				probs = torch.sigmoid(raw_logit)
				prediction = (torch.sigmoid(raw_logit) > 0.5).long()
				total_predictions.extend(prediction.view(-1).tolist())
				total_test_labels.extend(label.view(-1).tolist())
				# print(total_test_labels)
				# print(total_predictions)
				
				
				correct += (prediction == label).sum().item()
				# print(prediction, label, correct)
				total += label.size(0)

			accuracy = 100 * correct / total
			mean_test_loss = sum_test_loss / total
			
		# print(f"Test Accuracy: {accuracy:.2f}%")
		self.model.train()  # Set model to train mode
		return(accuracy, mean_test_loss, total_predictions, total_test_labels, probs)
	
	def regress(self):
		""" Evaluate the model on the test set """
		self.model.eval()  # Set model to evaluation mode
		total_r2 = []
		total=0
		sum_test_loss = 0
		total_predictions = []
		total_test_labels = []
		with torch.no_grad():
			for region, target in self.test_loader:
				
				
				# Move data to device
				region, target = region.to(self.device), target.to(self.device).to(torch.float)

				#Compute the raw logits
				raw_logit = self.model(region)
				test_loss = self.lossfun(raw_logit, target)
				sum_test_loss += test_loss.detach().item()
				# print(test_loss)
				# print(raw_logit)
				# print(raw_logit.shape, label.shape)

				# prediction = F.relu(raw_logit)
				# prediction = F.leaky_relu(raw_logit)
				prediction = raw_logit

				
				total_predictions.extend(prediction.view(-1).tolist())
				total_test_labels.extend(target.view(-1).tolist())
				# print(total_test_labels)
				# print(total_predictions)
				total+=1
				
				
				
				# print(prediction, label, correct)
				
					

				r2 = r2_score(target.cpu(), prediction.cpu())
				total_r2.append(r2)

			mean_test_loss = sum_test_loss / total
			
		# print(f"Test Accuracy: {accuracy:.2f}%")
		self.model.train()  # Set model to train mode
		return(r2, mean_test_loss, total_predictions, total_test_labels)
      
#################################################################################################################
#################################################################################################################

################## This is a simple trainer. It takes a model that outputs only a single logit ####################
# It can be used both for classsification and regression
#####################################################################
class simple_trainer_with_embed:
	def __init__(self, model, train_loader, test_loader,
					epochs=5, lr=0.001, weight_decay = 0, stopper_delta = 0.01, mode = 'classify', return_probs = False, use_early_stopping = True):
  

		self.model = model
		self.train_loader = train_loader 
		self.test_loader = test_loader
		self.epochs = epochs
		self.optimizer = torch.optim.AdamW(model.parameters(),lr, weight_decay=weight_decay) # Define the optimizer
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)  # Move model to GPU if available
		self.stopper_delta = stopper_delta
		self.early_stoper = EarlyStopping(patience=5, min_delta=self.stopper_delta, restore_best_weights=True,accuracy=True)
		self.use_early_stopping = use_early_stopping
		self.mode = mode
		self.return_probs = return_probs

		if self.mode == 'classify':
			self.lossfun = nn.BCEWithLogitsLoss() #Binary cross entropy loss for classification tasks
		elif self.mode == 'regress':
			self.lossfun = nn.MSELoss() 
			print('regression')
		


	def train(self):
		""" Train the model over multiple epochs """
		losses = []
		test_accuracies = []
		epoch_mean_batch_acc = []
		epoch_mean_test_losses = []

		# Loop over epochs
		for epoch in tqdm(range(self.epochs)):


			running_loss = 0.0
			self.model.train()  # Set model to training mode

			# Loop over minibatches in the train loader
			for hm in self.train_loader:
				regions, labels = hm

				regions, labels = regions.to(self.device), labels.to(self.device).to(torch.float) #Move data to device
				


				raw_logits, xembed = self.model(regions)



				## Important: The loss in binary classification is compouted from the raw logit and the tru label. Not the predicted label and the true label

				self.optimizer.zero_grad()  # Clear gradients from previous loops
				loss = (self.lossfun(raw_logits, labels))
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()
				
				 
				
				# batch_accuracy = self.batch_evaluate(X=regions, y=labels, sequence=sequence)
				# batch_acc.append(batch_accuracy)
				




			# print('classification')
			if self.mode == 'classify':
				
				epoch_acc_for_test, test_loss, total_predictions, total_test_labels, probs = self.evaluate()        
			elif self.mode == 'regress':
				epoch_acc_for_test, test_loss, total_predictions, total_test_labels = self.regress()
			
			# print(epoch_acc_for_test)

			model_weights = self.model.state_dict()
			if self.use_early_stopping:
				self.early_stoper(epoch_acc_for_test, self.model)
				if self.early_stoper.early_stop:
					model_weights = self.early_stoper.best_model_state
					break
                  
			# Append the sum of losses to the list that stores the loss from each epoch
			losses.append(running_loss / self.train_loader.batch_size)

			test_accuracies.append(epoch_acc_for_test)
			epoch_mean_test_losses.append(test_loss)
			
		if self.return_probs:
			return(test_accuracies, epoch_mean_batch_acc, losses, epoch_mean_test_losses, self.model, total_predictions, total_test_labels, probs, xembed)
		else:

			return(test_accuracies, epoch_mean_batch_acc, losses, epoch_mean_test_losses, self.model, total_predictions, total_test_labels, xembed)

			# return(losses)

				
	def evaluate(self):

		""" Evaluate the model on the test set """
		self.model.eval()  # Set model to evaluation mode

		correct = 0
		total = 0
		sum_test_loss = 0
		total_predictions = []
		total_test_labels = []
		with torch.no_grad():
			for region, label in self.test_loader:
				
				# Move data to device
				region, label = region.to(self.device), label.to(self.device).to(torch.float)

				#Compute the raw logits
				raw_logit, xembed = self.model(region)
				test_loss = self.lossfun(raw_logit, label)
				sum_test_loss += test_loss.detach().item()
				# print(test_loss)
				# print(raw_logit)
				# print(raw_logit.shape, label.shape)
                        
                

				probs = torch.sigmoid(raw_logit)
				prediction = (torch.sigmoid(raw_logit) > 0.5).long()
				total_predictions.extend(prediction.view(-1).tolist())
				total_test_labels.extend(label.view(-1).tolist())
				# print(total_test_labels)
				# print(total_predictions)
				
				
				correct += (prediction == label).sum().item()
				# print(prediction, label, correct)
				total += label.size(0)

			accuracy = 100 * correct / total
			mean_test_loss = sum_test_loss / total
			
		# print(f"Test Accuracy: {accuracy:.2f}%")
		self.model.train()  # Set model to train mode
		return(accuracy, mean_test_loss, total_predictions, total_test_labels, probs)
	
	def regress(self):
		""" Evaluate the model on the test set """
		self.model.eval()  # Set model to evaluation mode
		total_r2 = []
		total=0
		sum_test_loss = 0
		total_predictions = []
		total_test_labels = []
		with torch.no_grad():
			for region, target in self.test_loader:
				
				# Move data to device
				region, target = region.to(self.device), target.to(self.device).to(torch.float)

				#Compute the raw logits
				raw_logit, xembed = self.model(region)
				test_loss = self.lossfun(raw_logit, target)
				sum_test_loss += test_loss.detach().item()
				# print(test_loss)
				# print(raw_logit)
				# print(raw_logit.shape, label.shape)

				# prediction = F.relu(raw_logit)
				# prediction = F.leaky_relu(raw_logit)
				prediction = raw_logit

				
				total_predictions.extend(prediction.view(-1).tolist())
				total_test_labels.extend(target.view(-1).tolist())
				# print(total_test_labels)
				# print(total_predictions)
				total+=1
				
				
				
				# print(prediction, label, correct)
				
					

				r2 = r2_score(target.cpu(), prediction.cpu())
				total_r2.append(r2)

			mean_test_loss = sum_test_loss / total
			
		# print(f"Test Accuracy: {accuracy:.2f}%")
		self.model.train()  # Set model to train mode
		return(r2, mean_test_loss, total_predictions, total_test_labels)


# A trainer that trains a model that takes two inputs. It can be usedd for the seqcnn where the first input is the histone modification data and the other one is the one hot encoded sequence data.
class seqCNN_trainer:
	def __init__(self, model, train_loader, test_loader,
					epochs=5, lr=0.001, weight_decay = 0, stopper_delta = 0.01, mode = 'classify', return_probs = False, use_early_stopping = True):
  

		self.model = model
		self.train_loader = train_loader 
		self.test_loader = test_loader
		self.epochs = epochs
		self.optimizer = torch.optim.AdamW(model.parameters(),lr, weight_decay=weight_decay) # Define the optimizer. Optimizer is the component that performs the parameter update using the gradients
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)  # Move model to GPU if available
		self.stopper_delta = stopper_delta
		self.early_stoper = EarlyStopping(patience=5, min_delta=self.stopper_delta, restore_best_weights=True,accuracy=True)
		self.use_early_stopping = use_early_stopping
		self.mode = mode
		self.return_probs = return_probs # If the mode is classify

		if self.mode == 'classify':
			self.lossfun = nn.BCEWithLogitsLoss() #Binary cross entropy loss for classification tasks
		elif self.mode == 'regress':
			self.lossfun = nn.MSELoss() 
			print('regression')
		


	def train(self):
		""" Train the model over multiple epochs """
		losses = []
		test_accuracies = []
		epoch_mean_batch_acc = []
		epoch_mean_test_losses = []

		# Loop over epochs
		for epoch in tqdm(range(self.epochs)):


			running_loss = 0.0 #running loss is the sum of losses over the batches in the current epoch
			self.model.train()  # Set model to training mode

			# Loop over minibatches in the train loader
			for hm in self.train_loader:
				histone_data, sequence_data, labels = hm

				#Move data to device
				histone_data, sequence_data, labels = histone_data.to(self.device), sequence_data.to(self.device), labels.to(self.device).to(torch.float) #Move data to device
				#Forward pass through the model
				raw_logits, xembed = self.model(histone_data, sequence_data)



				## Important: The loss in binary classification is compouted from the raw logit and the tru label. Not the predicted label and the true label

				self.optimizer.zero_grad()  # Clear gradients from previous loops
				loss = (self.lossfun(raw_logits, labels))
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()
				
				 
				
				# batch_accuracy = self.batch_evaluate(X=regions, y=labels, sequence=sequence)
				# batch_acc.append(batch_accuracy)
				


										##########################################
										### EVALUATE THE MODEL ON THE TEST SET ###
										##########################################
			if self.mode == 'classify':
				
				epoch_acc_for_test, test_loss, total_predictions, total_test_labels, probs = self.evaluate()        
			elif self.mode == 'regress':
				epoch_acc_for_test, test_loss, total_predictions, total_test_labels = self.regress()
			
			# print(epoch_acc_for_test)
										###############################################
										### CHECK FOR ACC CHANGE TO STOIP THE MODEL ###
										###############################################
			model_weights = self.model.state_dict()
			if self.use_early_stopping:
				self.early_stoper(epoch_acc_for_test, self.model)
				if self.early_stoper.early_stop:
					model_weights = self.early_stoper.best_model_state
					break
                  
			# Append the sum of losses to the list that stores the loss from each epoch
			losses.append(running_loss / self.train_loader.batch_size)

			test_accuracies.append(epoch_acc_for_test)
			epoch_mean_test_losses.append(test_loss)
			
		if self.return_probs:
			return(test_accuracies, epoch_mean_batch_acc, losses, epoch_mean_test_losses, self.model, total_predictions, total_test_labels, probs, xembed)
		else:

			return(test_accuracies, epoch_mean_batch_acc, losses, epoch_mean_test_losses, self.model, total_predictions, total_test_labels, xembed)

			# return(losses)

				
	def evaluate(self):

		""" Evaluate the model on the test set """
		self.model.eval()  # Set model to evaluation mode. Evaluation mode shuts of batch normalization and dropout

		correct = 0
		total = 0
		sum_test_loss = 0
		total_predictions = []
		total_test_labels = []
		with torch.no_grad():
			for histone_data, sequence_data, labels in self.test_loader:
				
				# Move data to device
				histone_data, sequence_data, label = histone_data.to(self.device), sequence_data.to(self.device), labels.to(self.device).to(torch.float)

				#Compute the raw logits
				raw_logit, xembed = self.model(histone_data, sequence_data)
				test_loss = self.lossfun(raw_logit, label)
				sum_test_loss += test_loss.detach().item()
				# print(test_loss)
				# print(raw_logit)
				# print(raw_logit.shape, label.shape)
                        
                

				probs = torch.sigmoid(raw_logit) #Compute the probabilities by passing the data through a sigmoid
				prediction = (torch.sigmoid(raw_logit) > 0.5).long() #With ta threshold of 0.5 compute the predicted classes
				total_predictions.extend(prediction.view(-1).tolist())
				total_test_labels.extend(label.view(-1).tolist())
				correct += (prediction == label).sum().item()
				# print(prediction, label, correct)
				total += label.size(0)

			accuracy = 100 * correct / total
			mean_test_loss = sum_test_loss / total
			
		# print(f"Test Accuracy: {accuracy:.2f}%")
		self.model.train()  # Set model to train mode
		return(accuracy, mean_test_loss, total_predictions, total_test_labels, probs)
	
	def regress(self):
		""" Evaluate the model on the test set """
		self.model.eval()  # Set model to evaluation mode
		total_r2 = []
		total=0
		sum_test_loss = 0
		total_predictions = []
		total_test_labels = []
		with torch.no_grad():
			for histone_data, sequence_data, target in self.test_loader:
				
				# Move data to device
				histone_data, sequence_data, target = histone_data.to(self.device), sequence_data.to(self.device), target.to(self.device).to(torch.float)

				#Compute the raw logits
				raw_logit, xembed = self.model(histone_data, sequence_data)
				test_loss = self.lossfun(raw_logit, target)
				sum_test_loss += test_loss.detach().item()
				# print(test_loss)
				# print(raw_logit)
				# print(raw_logit.shape, label.shape)

				# prediction = F.relu(raw_logit)
				# prediction = F.leaky_relu(raw_logit)
				prediction = raw_logit

				
				total_predictions.extend(prediction.view(-1).tolist())
				total_test_labels.extend(target.view(-1).tolist())
				# print(total_test_labels)
				# print(total_predictions)
				total+=1
				
				
				
				# print(prediction, label, correct)
				
					

				r2 = r2_score(target.cpu(), prediction.cpu())
				total_r2.append(r2)

			mean_test_loss = sum_test_loss / total
			
		# print(f"Test Accuracy: {accuracy:.2f}%")
		self.model.train()  # Set model to train mode
		return(r2, mean_test_loss, total_predictions, total_test_labels)