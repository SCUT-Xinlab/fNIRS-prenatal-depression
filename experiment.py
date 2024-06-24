import torch
import numpy as np
import utils
import os
import wandb
from tqdm import tqdm
from dataloader import get_dataloader
from config import args
from networks import triFNRIS
from pretrain import PretrainStage
from sklearn.metrics import roc_auc_score
import numpy
# from network.bi_emotional import siam_2emo_psnet


def criterion(y: torch.tensor, y_hat: torch.tensor):
    assert y.shape == y_hat.shape, f"y.shape is {y.shape}, while y_hat.shape is {y_hat.shape}"
    with torch.no_grad():
        _, predicted = torch.max(y_hat, 1)
        _, groundtruth = torch.max(y, 1)
        TP = ((predicted == 1) & (groundtruth == 1)).sum().float().item()  # True Positive
        FP = ((predicted == 1) & (groundtruth == 0)).sum().float().item()  # False Positive
        FN = ((predicted == 0) & (groundtruth == 1)).sum().float().item()  # False Negative
        TN = ((predicted == 0) & (groundtruth == 0)).sum().float().item()  # True Negative
        TPR = TP / (TP + FN) if TP else 0  # True Positive Rate
        FPR = FP / (FP + TN) if FP else 0  # False Positive Rate
        FNR = FN / (TP + FN) if FN else 0  # False Negative Rate
        TNR = TN / (TN + FP) if TN else 0  # True Negative Rate
        confidence = (y * y_hat).sum().item() / y.shape[0]
        y_true = groundtruth.to("cpu").numpy()
        _y = y_hat[:, 1]
        y_scores = _y.to("cpu").numpy()
        if 1 in y_true and 0 in y_true:
            auc = roc_auc_score(y_true, y_scores)
        else:
            auc = 1
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP) if TP else 0
        recall = TP / (TP + FN) if TP else 0
        F1_score = 2.0 / (1.0 / (precision + args.eps) + 1.0 / (recall + args.eps) + args.eps)
        return TPR, FPR, confidence, accuracy, precision, recall, F1_score, auc

def AUC(FPR: list, TPR: list):
    with torch.no_grad():
        sorted_indices = sorted(range(len(FPR)), key=lambda i: FPR[i])
        FPR = [FPR[index] for index in sorted_indices]
        TPR = [TPR[index] for index in sorted_indices]
        FPR = np.concatenate(([0], FPR, [1]))
        TPR = np.concatenate(([0], TPR, [1]))
        AUC = np.trapz(TPR, FPR)
        return AUC
    
def data_aumentation(input: torch.tensor):
    return input



class Aumentation:
    def __init__(self, mean: float, std: float, step_size: int, gamma: float, max_std: float = 0.9, min_std: float = 0.001) -> None:
        self.mean = mean
        self.std = std
        self.step_size = step_size
        self.gamma = gamma
        self.max_std = max_std
        self.min_std = min_std
        self.step = 0
    
    def __call__(self, data: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            data = data + torch.randn(data.shape) * self.std + self.mean
        return data
        
    def step(self):
        self.step += 1
        if self.step >= self.step_size:
            self.std *= self.gamma
            if self.std > self.max_std:
                self.std = self.max_std
            if self.std < self.min_std:
                self.std = self.min_std
            self.step = 0
    
    



class Experiment:
    def __init__(self, k: int, epoch: int, num_classes: int, batch_size: int) -> None:
        self.k = k
        self.num_epochs = epoch
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.__init_dataloaders__()
        self.__init_aumentation__()
        self.best_auc = 0
        
    
    def __init_dataloaders__(self):
        self.train_dataloaders, self.test_dataloaders = get_dataloader(self.num_classes, self.batch_size, self.k)
        
        
    def __init_models__(self, k):
        if args.model_name == "ours":
            self.model = triFNRIS(encoder_name=args.encoder_name, decoder_name=args.decoder_name, header_name=args.header_name,
                              input_channels=args.input_size, output_channels=args.num_classes,embedding_dim=args.embedding_dim,
                              dropout=args.dropout_rate).to(args.device)
        elif args.model_name == "bi-emotional":
            # self.model = siam_2emo_psnet().to(args.device)
            raise NotImplementedError
        
        
        
        
        if args.pretrain:
            self.pretrain = PretrainStage(input_channels=args.input_size, output_channels=args.embedding_dim, 
                                          list_channels=args.TCN_list_channels, hidden_size=args.TCN_hidden_size, 
                                          kernel_size=args.TCN_kernel_size, dropout=args.dropout_rate,
                                          dataset=self.train_dataloaders[k], num_epochs=args.pretrain_epoch, patch_size=args.patch_size,
                                          prediction_length=args.prediction_length)
            self.pretrain.train()
            self.model.hap_encoder = self.model.sad_encoder = self.model.rs_encoder = self.pretrain.model.encoder
            
    
    def __init_loss_function__(self):
        loss_function_name = args.loss_function_name
        weights = torch.tensor(args.criterion_weights, dtype=torch.float).to(args.device)
        if loss_function_name == "BCE":
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            raise RuntimeError(f"no loss function names {loss_function_name}")
        
    def __init_optimizer__(self):
        optimizer_name = args.optimizer_name
        params = self.model.parameters()
        lr = args.init_lr
        betas = (args.beta_0, args.beta_1)
        eps = args.eps
        weight_decay = args.weight_decay
        momentum = args.momentum
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False)
        elif optimizer_name == "AMSGrad":
            self.optimizer = torch.optim.Adam(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=True)
        elif optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise RuntimeError(f"no optimizer names {optimizer_name}")
        
    def __init_scheduler__(self):
        if args.is_use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=args.optimizer_step_size, gamma=args.optimizer_gamma)
            
    def __init_aumentation__(self):
        if args.is_use_aumentation:
            self.aumentation = Aumentation(mean=args.aumentation_init_mean, std=args.aumentation_init_std, 
                                           step_size=args.aumentation_step_size, gamma=args.aumentation_gamma)
        
    def run(self):
        for k in tqdm(list(range(self.k))):
            self.__init_models__(k)
            self.__init_loss_function__()
            self.__init_optimizer__()
            self.__init_scheduler__()
            utils.log(f"+++++++++++ START {k}th folds ++++++++++")
            self.iteration(k)
            utils.log(f"+++++++++++ START {k}th folds ++++++++++")
        # utils.plot([os.path.join(args.csv_path, f"{i}.csv") for i in range(self.k)], os.path.join(args.logs_path, "result.png"), multidata=True)
    
    def iteration(self, k):
        exp_name = args.experiment_name + f"-{k}th-fold"
        if args.is_use_wandb:
            wandb.init(project=args.project_name, name=exp_name, save_code=True)
            wandb.watch(self.model, log="all")
        for self.epoch in range(self.num_epochs):
            self.trainTPR = []
            self.trainFPR = []
            self.testTPR = []
            self.testFPR = []
            # train
            train_loss, train_accuracy, train_auc, train_confidence, train_precision, train_recall, train_F1_score = self.train(k)
            # test
            test_loss, test_accuracy, test_auc, test_confidence, test_precision, test_recall, test_F1_score = self.test(k)
            # log
            utils.log(f"### epoch {self.epoch + 1:03} / {self.num_epochs:03} ### train_loss = {train_loss:.4f}, " +
                      f"train_confidence = {train_confidence * 100 :06.2f} %, test_loss = {test_loss:.4f}, test_confidence = {test_confidence * 100 :06.2f} %")
            if args.is_use_wandb:
                wandb.log({
                    "epoch": self.epoch,
                    "train_loss": train_loss, "test_loss": test_loss,
                    "train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                    "train_auc": train_auc, "test_auc": test_auc,
                    "train_precision": train_precision, "test_precision": test_precision,
                    "train_recall": train_recall, "test_recall": test_recall,
                    "trian_F1_score": train_F1_score, "test_F1_score": test_F1_score
                })
            result = {
                "epoch": [self.epoch, self.epoch],
                "loss": [train_loss, test_loss],
                "accuracy": [train_accuracy, test_accuracy],
                "auc": [train_auc, test_auc],
                "precision": [train_precision, test_precision],
                "recall": [train_recall, test_recall],
                "F1_score": [train_F1_score, test_F1_score],
                "hue": ["train", "test"]
            }
            utils.write_csv(result, os.path.join(args.csv_path, f"{k}.csv"))
            
        
        if args.is_use_wandb:
            wandb.config.update(args)
            wandb.save(os.path.join(args.logs_path, f"model-{k}th-fold"))
            wandb.finish()
        utils.plot(os.path.join(args.csv_path, f"{k}.csv"), os.path.join(args.logs_path, f"{k}.png"))
    
    def train(self, k):
        self.model.train()
        train_loss = 0
        total = 0
        TPR_mean, FPR_mean, confidence_mean, accuracy_mean, precision_mean, recall_mean, F1_score_mean, auc_mean = (None,) * 8
        for batch in self.train_dataloaders[k]:
            if args.is_use_extracted_features:
                hap, sad, rs, hap_feat, sad_feat, rs_feat, y = batch
            else:
                hap, sad, rs, y = batch
            # ========== START 1. Data Aumentation ==========
            if args.is_use_aumentation:
                hap = self.aumentation(hap)
                sad = self.aumentation(sad)
                rs = self.aumentation(rs)
            # ========== END 1. ==========
            
            # ========== START 2. Patchfy ==========
            def patch(data: torch.tensor) -> torch.tensor:
                if data.shape[2] % args.patch_size != 0:
                    raise RuntimeError(f"Sequence length {data.shape[2]} cannot be divided by patch size {args.patch_size}!")
                data = data.view(data.shape[0], data.shape[1], int(data.shape[2] / args.patch_size), args.patch_size)
                return data.mean(dim=3)
            if args.is_patchfy:
                hap = patch(hap)
                sad = patch(sad)
                rs = patch(rs)
            # ========== END 2. ==========
            
            # ========== START 3. Move data to device ==========
            hap = hap.to(args.device)
            sad = sad.to(args.device)
            rs = rs.to(args.device)
            if args.is_use_extracted_features:
                hap_feat = hap_feat.to(args.device)
                sad_feat = sad_feat.to(args.device)
                rs_feat = rs_feat.to(args.device)
            y = y.to(args.device)
            # ========== END 3. ==========
            
            # ========== START 4. Predict and Backward =========
            self.optimizer.zero_grad()
            if args.is_use_extracted_features:
                y_hat = self.model([hap, sad, rs, hap_feat, sad_feat, rs_feat])
            else:
                y_hat = self.model([hap, sad, rs])
            loss = self.loss_function(y_hat, y)
            train_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            loss.backward()
            self.optimizer.step()
            # ========== END 4. ==========
            
            # ========== 5. Evaluation ==========
            TPR, FPR, confidence, accuracy, precision, recall, F1_score, auc = criterion(y, y_hat)
            TPR_mean = utils.mean(TPR_mean, TPR)
            FPR_mean = utils.mean(FPR_mean, FPR)
            confidence_mean = utils.mean(confidence_mean, confidence)
            accuracy_mean = utils.mean(accuracy_mean, accuracy)
            precision_mean = utils.mean(precision_mean, precision)
            recall_mean = utils.mean(recall_mean, recall)
            F1_score_mean = utils.mean(F1_score_mean, F1_score)
            auc_mean = utils.mean(auc_mean, auc)
        loss = train_loss / total
        self.trainTPR.append(TPR_mean)
        self.trainFPR.append(FPR_mean)
        # ========== END 5. ==========
        return loss, accuracy_mean, auc_mean, confidence_mean, precision_mean, recall_mean, F1_score_mean
    
    def test(self, k):
        self.model.eval()
        test_loss = 0
        total = 0
        TPR_mean, FPR_mean, confidence_mean, accuracy_mean, precision_mean, recall_mean, F1_score_mean, auc_mean = (None,) * 8
        with torch.no_grad():
            for batch in self.test_dataloaders[k]:
                if args.is_use_extracted_features:
                    hap, sad, rs, hap_feat, sad_feat, rs_feat, y = batch
                else:
                    hap, sad, rs, y = batch
                # ========== START 1. Move data to device ==========
                hap = hap.to(args.device)
                sad = sad.to(args.device)
                rs = rs.to(args.device)
                if args.is_use_extracted_features:
                    hap_feat = hap_feat.to(args.device)
                    sad_feat = sad_feat.to(args.device)
                    rs_feat = rs_feat.to(args.device)
                y = y.to(args.device)
                # ========== END 1. =========
                
                # ========== START 2. Patchfy ==========
                def patch(data: torch.tensor) -> torch.tensor:
                    if data.shape[2] % args.patch_size != 0:
                        raise RuntimeError(f"Sequence length {data.shape[2]} cannot be divided by patch size {args.patch_size}!")
                    data = data.view(data.shape[0], data.shape[1], int(data.shape[2] / args.patch_size), args.patch_size)
                    return data.mean(dim=3)
                if args.is_patchfy:
                    hap = patch(hap)
                    sad = patch(sad)
                    rs = patch(rs)
                # ========== END 2. ==========

                # ========== START 3. Predict =========
                if args.is_use_extracted_features:
                    y_hat = self.model([hap, sad, rs, hap_feat, sad_feat, rs_feat])
                else:
                    y_hat = self.model([hap, sad, rs])
                loss = self.loss_function(y_hat, y)
                test_loss += loss.item() * y.shape[0]
                total += y.shape[0]
                # ========== END 3. ==========
                
                # ========== 4. Evaluation ==========
                TPR, FPR, confidence, accuracy, precision, recall, F1_score, auc = criterion(y, y_hat)
                TPR_mean = utils.mean(TPR_mean, TPR)
                FPR_mean = utils.mean(FPR_mean, FPR)
                confidence_mean = utils.mean(confidence_mean, confidence)
                accuracy_mean = utils.mean(accuracy_mean, accuracy)
                precision_mean = utils.mean(precision_mean, precision)
                recall_mean = utils.mean(recall_mean, recall)
                F1_score_mean = utils.mean(F1_score_mean, F1_score)
                auc_mean = utils.mean(auc_mean, auc)
            loss = test_loss / total
            self.testTPR.append(TPR_mean)
            self.testFPR.append(FPR_mean)
            # ========== END 4. ==========
            
            if auc >= self.best_auc:
                self.best_auc = auc
                torch.save(self.model.state_dict(), os.path.join(args.model_path, f"model-{k}th-fold.5h"))
            return loss, accuracy_mean, auc, confidence_mean, precision_mean, recall_mean, F1_score_mean