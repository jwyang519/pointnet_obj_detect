import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
import torch


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x, l3_points



class get_loss(nn.Module):
    def __init__(self, gamma=1.5, alpha=None):
        super(get_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred, target, trans_feat=None):
        # pred is already log-softmax output, convert to probabilities
        pred_exp = pred.exp()
        
        # Get the probability for the target class
        target_probs = pred_exp.gather(1, target.unsqueeze(1))
        
        # Calculate focal weight with reduced gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Calculate cross entropy with softer class weights if provided
        if self.alpha is not None:
            # Soften the class weights by taking the square root
            softened_weights = torch.sqrt(self.alpha)
            ce_loss = F.nll_loss(pred, target, weight=softened_weights, reduction='none')
        else:
            ce_loss = F.nll_loss(pred, target, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight.squeeze() * ce_loss
        
        return focal_loss.mean()
