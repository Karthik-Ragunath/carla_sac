import torch
import torch.nn as nn
import torch.nn.functional as F

# clamp bounds for Std of action_log
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0

class EnsembleActor(nn.Module):
    def __init__(self, rgb_image_model, bounding_box_image_model=None, merge_layer=True, add_feature_vector=False):
        super(EnsembleActor, self).__init__()
        self.rgb_image_model = rgb_image_model
        if merge_layer:
            self.bounding_box_image_model = bounding_box_image_model
            self.merge_layer_images = nn.Linear(2000, 1024)
            self.layer_1 = nn.Linear(1024, 256)
            self.layer_2 = nn.Linear(256, 128)
            self.layer_3 = nn.Linear(128, 64)
            self.layer_4 = nn.Linear(64, 32)
            if add_feature_vector:
                self.merge_layer_features = nn.Linear(46, 12)
            else:
                self.layer_5 = nn.Linear(32, 12)
        else:
            self.layer_1 = nn.Linear(1000, 512)
            self.layer_2 = nn.Linear(512, 256)
            self.layer_3 = nn.Linear(256, 128)
            self.layer_4 = nn.Linear(128, 32)
            self.layer_5 = nn.Linear(32, 12)
        
        self.actor_mean_layer_1 = nn.Linear(12, 4)
        self.actor_mean_layer_2 = nn.Linear(4, 2)

        self.actor_std_layer_1 = nn.Linear(12, 4)
        self.actor_std_layer_2 = nn.Linear(4, 2)
        
    def forward(self, rgb_input, bounding_box_input=None, feature_vector=None, merge_layer=True):
        rgb_features = self.rgb_image_model(rgb_input)
        if merge_layer:
            bounding_box_image_features = self.bounding_box_image_model(bounding_box_input)
            merged_image_features = torch.cat((rgb_features, bounding_box_image_features), dim=1)
            x = self.merge_layer_images(F.relu(merged_image_features))
            x = self.layer_1(F.relu(x))
            x = self.layer_2(F.relu(x))
            x = self.layer_3(F.relu(x))
            x = self.layer_4(F.relu(x))
            if feature_vector:
                x = torch.cat((x, feature_vector), dim=1)
                x = self.merge_layer_features(F.relu(x))
            else:
                x = self.layer_5(F.relu(x))
        else:
            x = self.layer_1(F.relu(rgb_features))
            x = self.layer_2(F.relu(x))
            x = self.layer_3(F.relu(x))
            x = self.layer_4(F.relu(x))
            x = self.layer_5(F.relu(x))
        
        act_mean = self.actor_mean_layer_1(F.relu(x))
        act_mean = self.actor_mean_layer_2(F.relu(act_mean))

        act_std = self.actor_std_layer_1(F.relu(x))
        act_std = self.actor_std_layer_2(F.relu(act_std))
        act_log_std = torch.clamp(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return act_mean, act_log_std

class EnsembleCritic(nn.Module):
    def __init__(self, rgb_image_model, bounding_box_image_model=None, merge_layer=True, add_feature_vector=False):
        super(EnsembleCritic, self).__init__()
        self.rgb_image_model = rgb_image_model
        if merge_layer:
            self.bounding_box_image_model = bounding_box_image_model
            self.merge_layer_images = nn.Linear(2000, 1024)
            self.layer_1 = nn.Linear(1024, 256)
            self.layer_2 = nn.Linear(256, 128)
            self.layer_3 = nn.Linear(128, 64)
            self.layer_4 = nn.Linear(64, 32)
            if add_feature_vector:
                self.merge_layer_features = nn.Linear(46, 12)
            else:
                self.layer_5 = nn.Linear(32, 12)
        else:
            self.layer_1 = nn.Linear(1000, 512)
            self.layer_2 = nn.Linear(512, 256)
            self.layer_3 = nn.Linear(256, 128)
            self.layer_4 = nn.Linear(128, 32)
            self.layer_5 = nn.Linear(32, 12)
        
        self.merge_layer_actions = nn.Linear(14, 256)

        self.q1_layer_1 = nn.Linear(256, 128)
        self.q1_layer_2 = nn.Linear(128, 64)
        self.q1_layer_3 = nn.Linear(64, 32)
        self.q1_layer_4 = nn.Linear(32, 4)
        self.q1_layer_5 = nn.Linear(4, 1)

        self.q2_layer_1 = nn.Linear(256, 128)
        self.q2_layer_2 = nn.Linear(128, 64)
        self.q2_layer_3 = nn.Linear(64, 32)
        self.q2_layer_4 = nn.Linear(32, 4)
        self.q2_layer_5 = nn.Linear(4, 1)

    def forward(self, rgb_input, bounding_box_input=None, actions=None, feature_vector=None, merge_layer=True):
        rgb_features = self.rgb_image_model(rgb_input)
        if merge_layer:
            bounding_box_image_features = self.bounding_box_image_model(bounding_box_input)
            merged_image_features = torch.cat((rgb_features, bounding_box_image_features), dim=1)
            x = self.merge_layer_images(F.relu(merged_image_features))
            x = self.layer_1(F.relu(x))
            x = self.layer_2(F.relu(x))
            x = self.layer_3(F.relu(x))
            x = self.layer_4(F.relu(x))
            if feature_vector:
                x = torch.cat((x, feature_vector), dim=1)
                x = self.merge_layer_features(F.relu(x))
            else:
                x = self.layer_5(F.relu(x))
        else:
            x = self.layer_1(F.relu(rgb_features))
            x = self.layer_2(F.relu(x))
            x = self.layer_3(F.relu(x))
            x = self.layer_4(F.relu(x))
            x = self.layer_5(F.relu(x))
        
        x = torch.cat((x, actions), dim=1)
        x = self.merge_layer_actions(F.relu(x))

        x1 = self.q1_layer_1(F.relu(x))
        x1 = self.q1_layer_2(F.relu(x1))
        x1 = self.q1_layer_3(F.relu(x1))
        x1 = self.q1_layer_4(F.relu(x1))
        x1 = self.q1_layer_5(F.relu(x1))

        x2 = self.q1_layer_1(F.relu(x))
        x2 = self.q2_layer_2(F.relu(x2))
        x2 = self.q2_layer_3(F.relu(x2))
        x2 = self.q2_layer_4(F.relu(x2))
        x2 = self.q2_layer_5(F.relu(x2))

        return x1, x2
