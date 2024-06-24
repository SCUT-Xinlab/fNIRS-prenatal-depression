import os
import datetime
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import json
from config import args

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log(info, level="INFO", path=None):
    path = path if path else args.logs_file
    now = datetime.datetime.now()
    s = f"[{level}][{now.hour:02}:{now.minute:02}:{now.second:02}] {info}"
    with open(path, "a", encoding="utf-8") as log:
        log.write(s + "\n")
    print(s)

def setup_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def mean(data1, data2):
    if (data1 is None) and (data2 is None):
        raise RuntimeError("mean(data1, data2), data1 is None and data2 is None!")
    if data1 is None:
        return data2
    if data2 is None:
        return data1
    return float(data1 + data2) / 2.0 

def write_csv(data, path):
    df = pd.DataFrame(data)
    if not os.path.isfile(path):
        df.to_csv(path, mode='a', index=False, header=True)
    else:
        df.to_csv(path, mode='a', index=False, header=False)
        
def plot(csv_path, png_path, multidata=False):
    if multidata:
        data = [pd.read_csv(path) for path in csv_path]
        data = pd.concat(data, axis=1)
    else:
        data = pd.read_csv(csv_path)
    sns.set()
    fig = plt.figure(figsize=(36, 20))
    for i, y in enumerate(["loss", "accuracy", "auc", "precision", "recall", "F1_score"]):
        ax = fig.add_subplot(2, 3, i + 1)
        sns.lineplot(data=data, x="epoch", y=y, hue="hue", ax=ax)
        ax.set_title(y)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(png_path)
    
def save_config(args, file):
    config = {key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, complex, list))}
    with open(file, "a") as f:
        json.dump(config, f)