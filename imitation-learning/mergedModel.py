import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class MyEnsemble(nn.Module):
    def __init__(self, semantic_model, uncertainty_model):
        super(MyEnsemble, self).__init__()
        self.semantic_model = semantic_model
        self.uncertainty_model = uncertainty_model
        self.mergeLayer = nn.Linear(2014, 1024)
        self.layer2 = nn.Linear(1024, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 3)
        
    def forward(self, semanticInput, uncertaintyInput, feature_vector):
        x1 = self.semantic_model(semanticInput)
        x2 = self.uncertainty_model(uncertaintyInput)
        x = torch.cat((x1, x2, feature_vector), dim=1)
        x = self.mergeLayer(F.relu(x))
        x = self.layer2(F.relu(x))
        x = self.layer3(F.relu(x))
        x = self.layer4(x)
        return x # THIS WILL RETURN THIS FUNCTION * 12 -> SO 128 BATCH SIZE

# Create models and load state_dicts    
# modelA = MyModelA()
# modelB = MyModelB()
# Load state dicts
# modelA.load_state_dict(torch.load(PATH))
# modelB.load_state_dict(torch.load(PATH))

# model = MyEnsemble(modelA, modelB)
# x1, x2 = torch.randn(1, 10), torch.randn(1, 20)
# output = model(x1, x2)
