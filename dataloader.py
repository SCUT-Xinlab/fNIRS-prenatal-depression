import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from config import args
from features_extraction import extract_features_nomalized



class MultiFeatureDataset(Dataset):
    def __init__(self, hap, sad, rs, y) -> None:
        super(MultiFeatureDataset, self).__init__()
        self.hap = hap
        self.sad = sad
        self.rs = rs
        self.y = y
        self.length = len(y)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        hap = self.hap[index]
        sad = self.sad[index]
        rs = self.rs[index]
        y = self.y[index]
        return hap, sad, rs, y

class ExtractedAndMultiFeatureDataset(Dataset):
    def __init__(self, hap, sad, rs, hap_feat, sad_feat, rs_feat, y) -> None:
        super(ExtractedAndMultiFeatureDataset, self).__init__()
        self.hap = hap
        self.sad = sad
        self.rs = rs
        self.hap_feat = hap_feat
        self.sad_feat = sad_feat
        self.rs_feat = rs_feat
        self.y = y
        self.length = len(y)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        hap = self.hap[index]
        sad = self.sad[index]
        rs = self.rs[index]
        hap_feat = self.hap_feat[index]
        sad_feat = self.sad_feat[index]
        rs_feat = self.rs_feat[index]
        y = self.y[index]
        return hap, sad, rs, hap_feat, sad_feat, rs_feat, y
    

    
def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # 1. select channels
    data = data[args.data_channels_list]
    # 2. select a sub sequence of length `args.data_length` from the end of the sequence
    data = data.tail(args.data_length - 1)
    # 3. add a zero to the start of the sequence
    zeros = pd.DataFrame({col: [0.0] for col in data.columns})
    data = pd.concat([zeros, data], ignore_index=True)
    return data

def read_data_of_one_object(path: str):
    if not os.path.exists(path):
        raise RuntimeError("Object not found!")
    # ========== START 1.Read csv data as pandas objects ===========
    hap = pd.read_csv(os.path.join(path, "hap.csv"))
    sad = pd.read_csv(os.path.join(path, "sad.csv"))
    rs = pd.read_csv(os.path.join(path, "rs.csv"))
    # ========== End 1. ==========
    
    # ========== START 2. Data preprocessing ==========
    hap = preprocessing(hap)
    sad = preprocessing(sad)
    rs = preprocessing(rs)
    # ========== End 2. ==========
    return hap, sad, rs
    

def get_dataloader(num_classes: int, batch_size: int, k: int, is_use_extracted_features: bool = True):
    trainset_dataloaders, testset_dataloaders = [], []
    for i in range(k):
        # ========== START 1. Make train dataset ==========
        train_root = os.path.join(args.data_root, str(i + 1), "train")
        train_objects = os.listdir(train_root)
        train_hap, train_sad, train_rs, train_y = [], [], [], []
        if is_use_extracted_features:
            train_hap_feat, train_sad_feat, train_rs_feat = [], [], []
        for object in train_objects:
            # ---------- 1.1 For each object, read happy, sad, rs csv file as DataFrame ----------
            path = os.path.join(train_root, object)
            hap, sad, rs = read_data_of_one_object(path)
            y = int(object[0])
            if is_use_extracted_features:
                hap_feat = extract_features_nomalized(hap)
                sad_feat = extract_features_nomalized(sad)
                rs_feat = extract_features_nomalized(rs)
            # ---------- 1.2 For each object, DataFrame -> torch.tensor ----------
            hap = torch.tensor(hap.values, dtype=args.input_data_type)
            sad = torch.tensor(sad.values, dtype=args.input_data_type)
            rs = torch.tensor(rs.values, dtype=args.input_data_type)
            if is_use_extracted_features:
                hap_feat = torch.tensor(hap_feat.values, dtype=args.input_data_type)
                sad_feat = torch.tensor(sad_feat.values, dtype=args.input_data_type)
                rs_feat = torch.tensor(rs_feat.values, dtype=args.input_data_type)
            # ---------- 1.3 For each object, append to list ----------
            train_hap.append(hap)
            train_sad.append(sad)
            train_rs.append(rs)
            train_y.append(y)
            if is_use_extracted_features:
                train_hap_feat.append(hap_feat)
                train_sad_feat.append(sad_feat)
                train_rs_feat.append(rs_feat)
        # ---------- 1.4 Make train_hap, train_sad, train_rs shape to [batch_size, num_channels, length_of_time_sequence] ----------
        train_hap = torch.stack(train_hap).permute(0, 2, 1)
        train_sad = torch.stack(train_sad).permute(0, 2, 1)
        train_rs = torch.stack(train_rs).permute(0, 2, 1)
        labels = train_y
        y = torch.tensor(train_y)
        train_y = torch.nn.functional.one_hot(y, num_classes).float()
        if is_use_extracted_features:
            train_hap_feat = torch.stack(train_hap_feat).permute(0, 2, 1)
            train_sad_feat = torch.stack(train_sad_feat).permute(0, 2, 1)
            train_rs_feat = torch.stack(train_rs_feat).permute(0, 2, 1)
        # ---------- 1.5 Make the weighted random sampler ----------
        weights = [1.0 / labels.count(label) for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        # ---------- 1.6 Get trainset DataLoader ----------
        if is_use_extracted_features:
            trainset_dataset = ExtractedAndMultiFeatureDataset(train_hap, train_sad, train_rs, train_hap_feat, train_sad_feat, train_rs_feat, train_y)
        else:
            trainset_dataset = MultiFeatureDataset(train_hap, train_sad, train_rs, train_y)
        trainset_dataloader = DataLoader(trainset_dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
        # ========== END 1. ==========
        
        # ========== START 2. Make test dataset ==========
        test_root = os.path.join(args.data_root, str(i + 1), "test")
        test_objects = os.listdir(test_root)
        print(test_objects)
        test_hap, test_sad, test_rs, test_y = [], [], [], []
        if is_use_extracted_features:
            test_hap_feat, test_sad_feat, test_rs_feat = [], [], []
        for object in test_objects:
            # ---------- 2.1 For each object, read happy, sad, rs csv file as DataFrame ----------
            path = os.path.join(test_root, object)
            hap, sad, rs = read_data_of_one_object(path)
            y = int(object[0])
            if is_use_extracted_features:
                hap_feat = extract_features_nomalized(hap)
                sad_feat = extract_features_nomalized(sad)
                rs_feat = extract_features_nomalized(rs)
            # ---------- 2.2 For each object, DataFrame -> torch.tensor ----------
            hap = torch.tensor(hap.values, dtype=args.input_data_type)
            sad = torch.tensor(sad.values, dtype=args.input_data_type)
            rs = torch.tensor(rs.values, dtype=args.input_data_type)
            if is_use_extracted_features:
                hap_feat = torch.tensor(hap_feat.values, dtype=args.input_data_type)
                sad_feat = torch.tensor(sad_feat.values, dtype=args.input_data_type)
                rs_feat = torch.tensor(rs_feat.values, dtype=args.input_data_type)
            # ---------- 2.3 For each object, append to list ----------
            test_hap.append(hap)
            test_sad.append(sad)
            test_rs.append(rs)
            test_y.append(y)
            if is_use_extracted_features:
                test_hap_feat.append(hap_feat)
                test_sad_feat.append(sad_feat)
                test_rs_feat.append(rs_feat)
        # ---------- 2.4 Make test_hap, test_sad, test_rs shape to [batch_size, num_channels, length_of_time_sequence] ----------
        test_hap = torch.stack(test_hap).permute(0, 2, 1)
        test_sad = torch.stack(test_sad).permute(0, 2, 1)
        test_rs = torch.stack(test_rs).permute(0, 2, 1)
        if is_use_extracted_features:
            test_hap_feat = torch.stack(test_hap_feat).permute(0, 2, 1)
            test_sad_feat = torch.stack(test_sad_feat).permute(0, 2, 1)
            test_rs_feat = torch.stack(test_rs_feat).permute(0, 2, 1)
        y = torch.tensor(test_y)
        test_y = torch.nn.functional.one_hot(y, num_classes).float()
        # ---------- 2.5 Get testset DataLoader ----------
        if is_use_extracted_features:
            testset_dataset = ExtractedAndMultiFeatureDataset(test_hap, test_sad, test_rs, test_hap_feat, test_sad_feat, test_rs_feat, test_y)
        else:
            testset_dataset = MultiFeatureDataset(test_hap, test_sad, test_rs, test_y)
        testset_dataloader = DataLoader(testset_dataset, batch_size=len(testset_dataset), shuffle=False)
        # ========== END 2. ==========
        trainset_dataloaders.append(trainset_dataloader)
        testset_dataloaders.append(testset_dataloader)
    return trainset_dataloaders, testset_dataloaders          