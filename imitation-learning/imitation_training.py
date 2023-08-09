import numpy as np
from mergedModel import MyEnsemble as fusionModel
import matplotlib.pyplot as plt
import xception
import torch.nn as nn
import torch.optim as optim
import torch
import pickle
from tqdm import tqdm
import random
import math
from sklearn.metrics import mean_squared_error
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from mergedModel import MyEnsemble as fusionModel
from statistics import mean
# command to run the logs on the Tensorboard:
# tensorboard dev upload --logdir="Jun20_14-21-17_AP4SBLJG3" --name="description"

#Variables
WIDTH = 300
HEIGHT = 300
EPOCHS = 400
MODEL_NAME = "Xception"
TRAINING_BATCH_SIZE = 256 #128 #32, 128

device = torch.device("cuda:2")
device2 = torch.device("cuda:2")
# device = torch.device("cpu")
torch.cuda.empty_cache() 

class imitation:
    def __init__(self):

        self.model = fusionModel(semantic_model=self.create_model(), uncertainty_model=self.create_model())#.to(device)
        count = 0
        for name, param in self.model.named_parameters():
            count += 1
            if param.requires_grad:
                if "model" not in name:
                #if "model" not in name or "conv3" in name or "bn3" in name or "bn4" in name or "conv4" in name or "last" in name:
                    print(name)
                else:
                    param.requires_grad = False

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0005)

    def create_model(self):
        return xception.xception(num_classes=1000, pretrained='imagenet')#.to(device)

    def getRMSE(self, y_tensor, yhat_tensor, action_index):
        y_list, yhat_list = [], []
        
        for i in range(len(y_tensor)):
            y_data = y_tensor[i, action_index].item()
            yhat_data = yhat_tensor[i, action_index].item()

            y_list.append(y_data)
            yhat_list.append(yhat_data)

        tensor_Yhat = torch.Tensor(yhat_list)
        tensor_Y = torch.Tensor(y_list)

        rmse = mean_squared_error(y_true=tensor_Y, y_pred=tensor_Yhat, squared=False)
        return rmse
        
    def truncate(self, value):
        str_value = str(value)
        output = ""
        for i in range(4):
            output += str_value[i]
        return float(output)

    def equals(self, y_tensor, yhat_tensor):
        
        if round((y_tensor.item()),2) == round(yhat_tensor.item(),2):
            return 1
        else:
            return 0
    
    def train(self):

        torch.cuda.empty_cache()
        criterion = nn.MSELoss()#.to(device)
        
        #uncertainty_states = []
        #semantic_states = []
        #supervised_labels = []

        actions_list = []
        images_list = []

        data_filename ='_out/imitation_training_data_100.pkl' #_out/imitation_data_512.pkl'
        images_filename ='_out/imitation_training_images_100.pkl' #'_out/imitation_images_512.pkl'

        with open(data_filename,'rb') as af:
            actions_list = pickle.load(af)

        with open(images_filename,'rb') as f:
            images_list = pickle.load(f)
        print("Dataset size: ", len(images_list))
        
        shuffled_list = []
        for i in range(len(images_list)):
            shuffled_list.append(i)
        random.shuffle(shuffled_list)

        TRAIN_SIZE =int(len(images_list) * 0.80) #0.80
        
        input_list = []

        for i in shuffled_list:

            semantic_state = (torch.from_numpy(images_list[i][0]).permute(2,0,1)/255)#.to(device)
            uncertainty_state = (torch.from_numpy(images_list[i][1]).permute(2,0,1)/255)#.to(device)

            #semantic_states.append(semantic_state) #Add the uncertainty/semantic segmented tuple
            #uncertainty_states.append(uncertainty_state)
            labels = torch.tensor(actions_list[i]) # 16 1 by 3 tensors (list of q value outputs
        
            input_list.append((semantic_state, uncertainty_state, labels)) 
            del semantic_state
            del uncertainty_state
            del labels
            #supervised_labels.append(torch.tensor(actions_list[i])) # 16 1 by 3 tensors (list of q value outputs
            # Set each entry that is big to 0 as we iterate to keep list size same
            images_list[i] = 0
            actions_list[i] = 0

        del images_list
        del actions_list
        
        print("data loaded into tensors")
        self.model.train()#.to(device2)
        
        tb = SummaryWriter()
        for lrate in [0.0005]:

            tb = SummaryWriter()
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lrate)
            self.model.train().to(device)

            for epoch in range(EPOCHS):
                random.shuffle(input_list)
                ##### Create Training Batch #####
                for i in range(0,TRAIN_SIZE,TRAINING_BATCH_SIZE):
                    epoch_itter = epoch * TRAIN_SIZE + i
                    self.optimizer.zero_grad()
                    x1 = torch.stack(list(zip(*input_list[i:i+int(TRAINING_BATCH_SIZE)]))[0]) #Semantic -> [[3x4]] -> 1x3x4 -> #
                    x2 = torch.stack(list(zip(*input_list[i:i+int(TRAINING_BATCH_SIZE)]))[1]) #states[i][1]#Uncertainty
                    y  = torch.stack(list(zip(*input_list[i:i+int(TRAINING_BATCH_SIZE)]))[2]) #batch size of 4 labels
                    y = y.type(torch.FloatTensor)
                    x1, x2, y  = x1.to(device), x2.to(device), y.to(device)

                    yhat = self.model(x1,x2)
                    #print("y shape:", y.shape, "yhat shape: ",yhat.shape)
                    #print("X1(RGB Batch Shape): ", x1.shape, "Inner Tensor Shape: ", x1[0].shape)

                    loss=criterion(yhat, y)
                    loss.backward()
                    self.optimizer.step()

                    del x1
                    del x2
                    torch.cuda.empty_cache()
                    print(f"Loss: epoch-itter: {epoch}-{i}", round(loss.item(), 3))

                    #### Calculating Training Accuracy ####
                    cor_steer = cor_throttle = cor_brake = 0
                    for j in range(len(y)):

                        cor_steer += self.equals(y[j,0],yhat[j,0])
                        cor_throttle += self.equals(y[j,1],yhat[j,1])
                        cor_brake += self.equals(y[j,2],yhat[j,2])

                    accuracy_steer = round(cor_steer/TRAINING_BATCH_SIZE, 3)
                    accuracy_throttle = round(cor_throttle/TRAINING_BATCH_SIZE, 3)
                    accuracy_brake = round(cor_brake/TRAINING_BATCH_SIZE, 3)
                    accuracy_avg = round((accuracy_steer + accuracy_throttle + accuracy_brake) / 3, 3)

                    rmse_steer = self.getRMSE(y, yhat, 0)
                    rmse_throttle = self.getRMSE(y, yhat, 1)
                    rmse_brake = self.getRMSE(y, yhat, 2)

                    del y
                    del yhat

                    #### Add Tensorboard Training Metrics ####
                    tb.add_scalar("Accuracy_avg", accuracy_avg, epoch_itter)
                    tb.add_scalar("Accuracy_steer", accuracy_steer, epoch_itter)
                    tb.add_scalar("Accuracy_throttle", accuracy_throttle, epoch_itter)
                    tb.add_scalar("Accuracy_brake", accuracy_brake, epoch_itter)

                    tb.add_scalar("Loss", loss, epoch_itter)

                    tb.add_scalar("RMSE", math.sqrt(loss), epoch_itter)
                    tb.add_scalar("Steer RMSE", rmse_steer, epoch_itter)
                    tb.add_scalar("Throttle RMSE", rmse_throttle, epoch_itter)
                    tb.add_scalar("Brake RMSE", rmse_brake, epoch_itter)



###### Validating the model
                print("Training complete, validating model")
                torch.cuda.empty_cache()
                VAL_TRAINING_BATCH_SIZE = TRAINING_BATCH_SIZE
                self.model.eval().to(device2)
                test_accuracy_tuple = []
                rmse_tuple = []
                # Turn off gradient since validation only
                with torch.no_grad():
                    # Iterate over epoch
                    for i in range(TRAIN_SIZE, len(input_list), VAL_TRAINING_BATCH_SIZE):
                        epoch_itter = epoch * TRAIN_SIZE + i
                        x1_test = torch.stack(list(zip(*input_list[i:i+int(VAL_TRAINING_BATCH_SIZE)]))[0])
                        x2_test = torch.stack(list(zip(*input_list[i:i+int(VAL_TRAINING_BATCH_SIZE)]))[1])
                        y_test  = torch.stack(list(zip(*input_list[i:i+int(VAL_TRAINING_BATCH_SIZE)]))[2])
                        y_test = y_test.type(torch.FloatTensor)
                        x1_test, x2_test, y_test  = x1_test.to(device2), x2_test.to(device2), y_test.to(device2)

                        # Get model predictions for validation
                        yhat_test = self.model(x1_test, x2_test)

                        del x1_test
                        del x2_test
                        torch.cuda.empty_cache()

                        test_correct_steer = test_correct_throttle = test_correct_brake = 0

                        for k in range(len(y_test)):
                            test_correct_steer += self.equals(y_test[k,0],yhat_test[k,0])
                            test_correct_throttle += self.equals(y_test[k,1], yhat_test[k,1])
                            test_correct_brake += self.equals(y_test[k,2], yhat_test[k,2])

                        test_accuracy_steer = round(test_correct_steer/VAL_TRAINING_BATCH_SIZE, 3)
                        test_accuracy_throttle = round(test_correct_throttle/VAL_TRAINING_BATCH_SIZE, 3)
                        test_accuracy_brake = round(test_correct_brake/VAL_TRAINING_BATCH_SIZE, 3)
                        test_accuracy_avg = round((test_accuracy_steer + test_accuracy_throttle + test_accuracy_brake) / 3, 3)
                        
                        test_accuracy_tuple.append((test_accuracy_steer,test_accuracy_throttle,test_accuracy_brake,test_accuracy_avg))

                        rmse_steer_validation = self.getRMSE(y_test, yhat_test, 0)
                        rmse_throttle_validation = self.getRMSE(y_test, yhat_test, 1)
                        rmse_brake_validation = self.getRMSE(y_test, yhat_test, 2)
    
                        rmse_tuple.append((rmse_steer_validation,rmse_throttle_validation,rmse_brake_validation))
                        # y_test = y_test.cpu().numpy()
                        # yhat_test = yhat_test.cpu().numpy()
                        
                        # #debugging
                        # if test_accuracy_steer == 0:
                        #     with open("_out/debugValidation_512.txt",'a') as f:
                        #         f.write("EPOCH ITER: " + str(epoch_itter)+"\n")
                        #         f.write("STEER Y\n")
                        #         f.write(str(y_test[0,0].item()))
                        #         f.write("\nSTEER YHAT\n")
                        #         f.write(str(yhat_test[0,0].item()))
                        #         f.write("\n\n-------\n\n")

                        del y_test
                        del yhat_test
                        torch.cuda.empty_cache()

                
                    test_acc_steer = mean(list(zip(*test_accuracy_tuple))[0])
                    test_acc_throttle = mean(list(zip(*test_accuracy_tuple))[1])
                    test_acc_brake = mean(list(zip(*test_accuracy_tuple))[2])
                    test_acc_average = mean(list(zip(*test_accuracy_tuple))[3])
                    tb.add_scalar("validation_accuracy_avg", test_acc_average, epoch)
                    tb.add_scalar("validation_accuracy_steer", test_acc_steer, epoch)
                    tb.add_scalar("validation_accuracy_throttle", test_acc_throttle, epoch)
                    tb.add_scalar("validation_accuracy_brake", test_acc_brake, epoch)
                    
                    rmse_val_steer = mean(list(zip(*rmse_tuple))[0])
                    rmse_val_throttle = mean(list(zip(*rmse_tuple))[1])
                    rmse_val_brake = mean(list(zip(*rmse_tuple))[2])
                    tb.add_scalar("Steer Validation RMSE", rmse_val_steer, epoch)
                    tb.add_scalar("Throttle Validation RMSE", rmse_val_throttle, epoch)
                    tb.add_scalar("Brake Validation RMSE", rmse_val_brake, epoch)
                    
                    if epoch % 50 == 0:
                        name = str("imitation_models/100k_3layers_newvalidation_nocurve_"+str(epoch)+"epochs_256batch.pt")
                        torch.save(self.model,name)
            tb.close()
            torch.save(self.model,f'imitation_models/final_100k_3layers_newvalidation_nocurve_400epochs_256batch.pt')  
            torch.cuda.empty_cache()

agent = imitation()
agent.train()
