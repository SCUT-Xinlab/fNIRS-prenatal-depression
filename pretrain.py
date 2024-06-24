import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.TCN
from torch.utils.data import DataLoader
from config import args
from dataloader import get_dataloader

class PretrainModel(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, list_channels: list, 
                 hidden_size: int, kernel_size: int, dropout: float, prediction_length: int) -> None:
        super(PretrainModel, self).__init__()
        self.prediction_length = prediction_length
        self.encoder = models.TCN.TCN(input_channels, output_channels, list_channels, hidden_size, kernel_size, dropout)
        self.fc1 = nn.Linear(output_channels, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, prediction_length * input_channels)
        self.fc4 = nn.Linear(output_channels, 32)
        self.fc5 = nn.Linear(32, 3)
        self.bn64 = nn.BatchNorm1d(64)
        self.bn32 = nn.BatchNorm1d(32)
        self.sequence_weights = nn.Parameter(torch.empty(3).normal_(mean=0, std=0.02), requires_grad=True)
        
    def combine(self, x):
        num_channels = int(x.shape[1] / 3)
        weights = F.softmax(self.sequence_weights, dim=0)
        HbO = weights[0] * x[:, 0: num_channels, :]
        HbR = weights[1] * x[:, num_channels: 2 * num_channels, :]
        HbT = weights[2] * x[:, 2 * num_channels: 3 * num_channels, :]
        x = HbO + HbR + HbT
        return x
        
    def forward(self, x):
        batch_size, num_channels = x.shape[0], x.shape[1]
        # ========== START 1. Mask the sequence ==========
        mask = torch.zeros((batch_size, num_channels, self.prediction_length), dtype=x.dtype).to(args.device)
        x = torch.concat([mask, x[:, :, :-self.prediction_length]], dim=2)
        # ========== END 1. ==========
        
        # ========== START 2. Combine 3 sequences into 1 weighted sequence ==========
        x = self.combine(x)
        # ========== END 2. ==========
        
        # ========== START 3. Compute the sequence embedding ==========
        embedding = self.encoder(x)
        # ========== END 3. ==========
        
        # ========== START 4. Predict the end of the sequence ==========
        x = self.bn64(F.relu(self.fc1(embedding)))
        x = self.bn64(F.relu(self.fc2(x)))
        x = self.fc3(x)
        predicted_x = x.view(batch_size, int(num_channels / 3), self.prediction_length)
        # ========== END 4. ==========
        
        # ========== START 5. Predict the class (hap, sad, or rs) of the sequence ==========
        x = self.bn32(F.relu(self.fc4(embedding)))
        x = self.fc5(x)
        classification = F.softmax(x, dim=1)
        # ========== END 5. ==========
        return predicted_x, classification
        
class PretrainStage:
    def __init__(self, input_channels: int, output_channels: int, list_channels: list, hidden_size: int, 
                 kernel_size: int, dropout: float, prediction_length: int, dataset: DataLoader, num_epochs: int, patch_size: int) -> None:
        self.model = PretrainModel(input_channels, output_channels, list_channels, hidden_size, kernel_size, dropout, prediction_length).to(args.device)
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.patch_size = patch_size
        self.predcition_length = prediction_length
        lr = args.init_lr
        betas = (args.beta_0, args.beta_1)
        eps = args.eps
        weight_decay = args.weight_decay
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False)
        self.MSELoss = nn.MSELoss()
        self.CELoss = nn.CrossEntropyLoss()
        
    def train(self):
        self.model.train()
        print("pretrain start!")
        for epoch in range(self.num_epochs):
            train_loss, total = 0, 0
            for batch in self.dataset:
                # ========== START 1. Read the data ==========
                if args.is_use_extracted_features:        
                    hap, sad, rs, hap_feat, sad_feat, rs_feat, y = batch
                else:
                    hap, sad, rs, y = batch
                # ========== END 1. ==========
                
                # ========== START 2. Patchfy ==========
                def patch(data: torch.tensor) -> torch.tensor:
                    if data.shape[2] % self.patch_size != 0:
                        raise RuntimeError(f"Sequence length {data.shape[2]} cannot be divided by patch size {self.patch_size}!")
                    data = data.view(data.shape[0], data.shape[1], int(data.shape[2] / self.patch_size), self.patch_size)
                    return data.mean(dim=3)
                hap = patch(hap).to(args.device)
                sad = patch(sad).to(args.device)
                rs = patch(rs).to(args.device)
                # ========== END 2. ==========
                
                # ========== START 3. Set labels and shuffle the data ==========
                batch_size = hap.shape[0]
                y = torch.tensor([0] * batch_size + [1] * batch_size + [2] * batch_size, dtype=torch.long)
                x = torch.concat([hap, sad, rs], dim=0)
                indecis = np.arange(x.size(0))
                np.random.shuffle(indecis)
                x = x[indecis]
                y = y[indecis]
                with torch.no_grad():
                    x_ground_truth = self.model.combine(x)[:, :, -self.predcition_length:]
                y = F.one_hot(y, 3).float().to(args.device)
                # ========== END 3. ==========
                
                # ========== START 4. Predict and backward ==========
                self.optimizer.zero_grad()
                predicted, classification = self.model(x)
                if args.pretrain == "reconstruction":
                    loss = self.MSELoss(x_ground_truth, predicted)
                elif args.pretrain == "classification":
                    loss = self.CELoss(classification, y)
                elif args.pretrain == "reconstruction-classification":
                    loss = self.MSELoss(x_ground_truth, predicted) + self.CELoss(classification, y)
                else:
                    raise RuntimeError(f"no pretrain methods name {args.pretrain}")
                train_loss += loss.item() * y.shape[0]
                total += y.shape[0]
                loss.backward()
                self.optimizer.step()
                # ========== END 4. ==========