import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
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
#from sklearn.metrics import mean_squared_error, r2_score
#import sklearn.model_selection import train_test_split
#import tiramisuModel.tiramisu as tiramisu
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from mergedModel import MyEnsemble as fusionModel

# command to run the logs on the Tensorboar:
# tensorboard dev upload --logdir="C:\Users\s1929247\Documents\Ali-Document\Computer Science\Project\imitation\runs\Jun20_14-21-17_AP4SBLJG3"

#Variables
MEMORY_FRACTION = 0.6
WIDTH = 300
HEIGHT = 300
EPOCHS = 300
MODEL_NAME = "Xception"
TRAINING_BATCH_SIZE = 512 #128 #32, 128

device = torch.device("cuda:2")
device2 = torch.device("cuda:2")
# device = torch.device("cpu")
torch.cuda.empty_cache() 

class imitation:
    def __init__(self):
        self.model = self.create_model()
       # self.model = fusionModel(semantic_model=self.create_model(), uncertainty_model=self.create_model())#.to(device)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if not "last" in name:
                    param.requires_grad = False
                else:
                    print(name) 

        #self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)

    def create_model(self):
        return xception.xception(num_classes=3, pretrained=False)#.to(device)

    # Convert a continous action into one of the buckets: 
    # -0.6, -0.3 , -0.015, -0.03, -0.045, 0.00, 0.015, 0.03, 0.045 , 0.3, 0.6
    def discretise(self, continuous_action):
        # By default we round to 2 places to discretise
        discrete_action = round(continuous_action, 2)

        if -1 <= continuous_action < -0.6:
            discrete_action = -0.6
            
        elif -0.6 <= continuous_action < -0.3:
             discrete_action = -0.3
            
        elif -0.3 <= continuous_action < -0.045:
             discrete_action = -0.045

        elif -0.045 <= continuous_action < -0.03:
            discrete_action = -0.03

        elif -0.03 <= continuous_action < -0.015:
            discrete_action = -0.015

        elif -0.045 <= continuous_action < -0.03:
            discrete_action = -0.03
        
        elif -0.03 <= continuous_action < 0:
            discrete_action = 0
        
        elif 0 <= continuous_action < 0.015:
            discrete_action = 0

        elif 0.015 <= continuous_action < 0.03:
            discrete_action = 0.015

        elif 0.03 <= continuous_action < 0.045:
            discrete_action = 0.03

        elif 0.045 <= continuous_action < 0.3:
            discrete_action = 0.045

        elif 0.3 <= continuous_action < 0.6:
            discrete_action = 0.3
            
        elif 0.6 <= continuous_action <= 1:
            discrete_action = 0.6

        return discrete_action
        
    def train(self):

        torch.cuda.empty_cache()
        torch.autograd.set_detect_anomaly(True)
        criterion = nn.MSELoss()#.to(device)

        semantic_states = []
        supervised_labels = []

        actions_list = []
        images_list = []

        with open('_out/imitation_training_data.pkl','rb') as af:
            actions_list = pickle.load(af)

        with open('_out/imitation_training_images.pkl','rb') as f:
            images_list = pickle.load(f)
        print(len(images_list))

        shuffled_list = []
        for i in range(len(images_list)):
            shuffled_list.append(i)
        random.shuffle(shuffled_list)

        TRAIN_SIZE =int(len(images_list) * 0.80) #0.80
        
        for i in shuffled_list:

            semantic_state = (torch.from_numpy(images_list[i][0]).permute(2,0,1)/255)#.to(device)
            #uncertainty_state = (torch.from_numpy(images_list[i][1]).permute(2,0,1)/255)#.to(device)

            semantic_states.append(semantic_state) #Add the uncertainty/semantic segmented tuple
            #uncertainty_states.append(uncertainty_state)
            
            del semantic_state
            #del uncertainty_state

            # if actions_list[i] < 0.3 && > 0.0 : supervised_labels.append(0.2)
            # 0.05679 -> <0.5 x<0.5 append 0.00 -> (0.2) 0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.2
            # TURNED OFF DISCRETIZE    #actions_list[i][0] = self.discretise(actions_list[i][0])
            supervised_labels.append(torch.tensor(actions_list[i])) # 16 1 by 3 tensors (list of q value outputs
            # Set each entry that is big to 0 as we iterate to keep list size same
            images_list[i] = 0
            actions_list[i] = 0

        del images_list
        del actions_list

        # images_list.clear()
        print("data loaded into tensors")
        self.model.train().to(device)
        #del self.loss
        # print('len semantics', len(semantic_states))
        
        tb = SummaryWriter()
        for lrate in [0.0005]:

            tb = SummaryWriter()
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lrate)
            self.model.train().to(device)

            for epoch in range(EPOCHS):
                
                for i in range(0,TRAIN_SIZE,TRAINING_BATCH_SIZE):

                    epoch_itter = epoch * TRAIN_SIZE + i
                    self.optimizer.zero_grad()
                    x1  = torch.stack((semantic_states[i:i+int(TRAINING_BATCH_SIZE)]))#Semantic -> [[3x4]] -> 1x3x4 -> #
                    y   = torch.stack(supervised_labels[i:i+int(TRAINING_BATCH_SIZE)]) #batch size of 4 labels
                    y   = y.type(torch.FloatTensor)
                    x1, y  = x1.to(device), y.to(device)

                    yhat = self.model(x1)
                    #print("y shape:", y.shape, "yhat shape: ",yhat.shape)

                    loss=criterion(yhat, y)
                    loss.backward()
                    self.optimizer.step()

                    del x1
                    torch.cuda.empty_cache()
                    print(f"Loss: epoch-itter: {epoch}-{i}", round(loss.item(), 3))

                    cor_steer = cor_throttle = cor_brake = 0

                    for j in range(len(y)):

                        cor_steer += (torch.round(y[j,0], decimals=2) == torch.round(yhat[j,0], decimals=2)).sum().item()
                        cor_throttle += (torch.round(y[j,1], decimals=2) == torch.round(yhat[j,1], decimals=2)).sum().item()
                        cor_brake += (torch.round(y[j,2], decimals=2) == torch.round(yhat[j,2], decimals=2)).sum().item()
                    del y
                    del yhat

                    accuracy_steer = round(cor_steer/TRAINING_BATCH_SIZE, 3)
                    accuracy_throttle = round(cor_throttle/TRAINING_BATCH_SIZE, 3)
                    accuracy_brake = round(cor_brake/TRAINING_BATCH_SIZE, 3)
                    accuracy_avg = round((accuracy_steer + accuracy_throttle + accuracy_brake) / 3, 3)

                    tb.add_scalar("Loss", loss, epoch_itter)
                    tb.add_scalar("RMSE", math.sqrt(loss), epoch_itter)
                    tb.add_scalar("Accuracy_steer", accuracy_steer, epoch_itter)
                    tb.add_scalar("Accuracy_throttle", accuracy_throttle, epoch_itter)
                    tb.add_scalar("Accuracy_brake", accuracy_brake, epoch_itter)
                    tb.add_scalar("Accuracy_avg", accuracy_avg, epoch_itter)

                    # print("Accuracy_steer: ", accuracy_steer, "Accuracy_throttle: ", accuracy_throttle, "Accuracy_brake: ", accuracy_brake, "Accuracy_avg: ", accuracy_avg)


######Validating the model
                print("Training complete, validating model")
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for i in range(TRAIN_SIZE,len(shuffled_list)):
                        self.model.eval().to(device2)
                        x1_test    = torch.unsqueeze(semantic_states[i], 0)#Semantic
                        y_test     = torch.unsqueeze(supervised_labels[i], 0) #batch size of 4 labels
                        x1_test, y_test  = x1_test.to(device2), y_test.to(device2)
                        #print("x1 test shape: ",x1_test.shape)
                        yhat_test = self.model(x1_test)

                        del x1_test
                        torch.cuda.empty_cache()

                        test_correct_steer = test_correct_throttle = test_correct_brake = 0
                        for k in range(len(y_test)):
                            test_correct_steer += (torch.round(y_test[k,0], decimals=2) == torch.round(yhat_test[k,0], decimals=2)).sum().item()
                            test_correct_throttle += (torch.round(y_test[k,1], decimals=2) == torch.round(yhat_test[k,1], decimals=2)).sum().item()
                            test_correct_brake += (torch.round(y_test[k,2], decimals=2) == torch.round(yhat_test[k,2], decimals=2)).sum().item()
                        test_accuracy_steer = round(test_correct_steer/len(y_test), 3)
                        test_accuracy_throttle = round(test_correct_throttle/len(y_test), 3)
                        test_accuracy_brake = round(test_correct_brake/len(y_test), 3)
                        test_accuracy_avg = round((test_accuracy_steer + test_accuracy_throttle + test_accuracy_brake) / 3, 3)
                        y_test = y_test.cpu().numpy()
                        yhat_test = yhat_test.cpu().numpy()
                        #mse = mean_squared_error(y_test, yhat_test)
                        #r_square = r2_score(y_test, yhat_test)
                        # print("validation_accuracy_steer: ",test_accuracy_steer,"validation_accuracy_throttle: ",test_accuracy_throttle,"validation_accuracy_brake: ",test_accuracy_brake)
                        # print("validation_accuracy_avg: ", test_accuracy_avg, "Mean Squared Error :", mse, "R^2 :",r_square)
                        tb.add_scalar("validation_accuracy_steer", test_accuracy_steer, epoch)
                        tb.add_scalar("validation_accuracy_throttle", test_accuracy_throttle, epoch)
                        tb.add_scalar("validation_accuracy_brake", test_accuracy_brake, epoch)
                        tb.add_scalar("validation_accuracy_avg", test_accuracy_avg, epoch)
                        #tb.add_scalar("validation_MSE", mse, epoch)
                        #tb.add_scalar("validation_R2", r_square, epoch)

                        del y_test
                        del yhat_test
                        #del mse
                       # del r_square
                        torch.cuda.empty_cache()

            tb.close()
            torch.save(self.model,f'imitation_models/singleRGB.pt')  
            torch.cuda.empty_cache()

agent = imitation()
agent.train()
