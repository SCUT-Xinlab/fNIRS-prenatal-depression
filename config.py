import os
import argparse
import datetime
import json
import torch

parser = argparse.ArgumentParser()
now = datetime.datetime.now()


# ========== Settings ==========
parser.add_argument("--is-use-cuda", type=bool, default=True, help="True if use cuda else False")
parser.add_argument("-random-seed", type=int, default=2023, help="random seed")
parser.add_argument("--eps", type=float, default=1e-08, help="epsilon")
parser.add_argument("--project-name", type=str, default="triFNRIS", help="name of the project")
parser.add_argument("--experiment-name", type=str, default=str(now), help="name of the experiment")
parser.add_argument("--data-root", type=str, default=os.path.join("data", "mothers", "3-folds"), help="root path of data")
parser.add_argument("--channels-name-file", type=str, default=os.path.join("data", "channels.json"), help="channels name json file")
parser.add_argument("--input-data-type-name", type=str, default="float", help="type of input data")
parser.add_argument("--is-use-wandb", type=bool, default=False, help="True if use wandb else False")
# ========== Data ==========
parser.add_argument("--data-channels", type=str, default="right_brain_HbT", help="channels name of the input data")
parser.add_argument("--fs", type=float, default=11, help="sampling frequency of the fNRIS sequence")
parser.add_argument("--is-use-extracted-features", type=bool, default=True, help="True if use extracted features else False")
parser.add_argument("--num-extracted-features", type=int, default=9, help="nums of the extracted features")
# ---------- Preprocessing ----------
parser.add_argument("--data-length", type=int, default=1500, help="cliped sequence length")




# ========== Train ==========
parser.add_argument("--pretrain", type=str, default=None, help="methods of pretrain") # "reconstruction-classification"
parser.add_argument("--pretrain-epoch", type=int, default=10, help="nums of epochs of pretrain")
parser.add_argument("--prediction-length", type=int, default=30, help="length of predicted sequence")
parser.add_argument("--patch-size", type=int, default=10, help="patch size")
parser.add_argument("--loss-function-name", type=str, default="BCE", help="name of the loss function")
parser.add_argument("--k-folds", type=int, default=3, help="nums of k-folds, must no more than 3")
parser.add_argument("--num-classes", type=int, default=2, help="nums of classes")
parser.add_argument("--num-epochs", type=int, default=250, help="nums of epoches")
parser.add_argument("--batch-size", type=int, default=8, help="batch size")
parser.add_argument("--criterion_weights", type=list, default=[1.0, 1.0], help="criterion weights")
parser.add_argument("--is-patchfy", type=bool, default=True, help="True if patchfy else False")
parser.add_argument("--use-sequence", type=str, default="hap-sad-rs", help="the sequence used")
# ---------- Optimizer ----------
parser.add_argument("--optimizer-name", type=str, default="Adam", help="name of optimizer")
parser.add_argument("--init-lr", type=float, default=0.001, help="initail learning rate")
parser.add_argument("--beta_0", type=float, default=0.9, help="beta_0 of optimizer")
parser.add_argument("--beta_1", type=float, default=0.999, help="beta_1 of optimizer")
parser.add_argument("--weight-decay", type=float, default=0.05, help="weight decay of optimizer")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum of optimizer")
# ---------- Scheduler ----------
parser.add_argument("--is-use-scheduler", type=bool, default=True, help="True if use scheduler else False")
parser.add_argument("--optimizer-step-size", type=int, default=10, help="step size of scheduler")
parser.add_argument("--optimizer-gamma", type=float, default=0.9, help="gamma of scheduler")
# ---------- Aumentation ----------
parser.add_argument("--is-use-aumentation", type=bool, default=True, help="True if use data aumentation eles False")
parser.add_argument("--aumentation-init-std", type=float, default=0.6, help="init standard or gaussain noise in data aumentation")
parser.add_argument("--aumentation-init-mean", type=float, default=0, help="init mean of gaussian noise in data aumentation")
parser.add_argument("--aumentation-step-size", type=int, default=10, help="step size of standard decrease in data aumentation")
parser.add_argument("--aumentation-gamma", type=float, default=0.9, help="decrease rate of standard in data aumentation")





# ========== Model ==========
parser.add_argument("--encoder-name", type=str, default="TCN", help="name of the encoder")
parser.add_argument("--decoder-name", type=str, default="MLP", help="name of the decoder")
parser.add_argument("--header-name", type=str, default="2_classes_MLP", help="name of the header")
parser.add_argument("--embedding-dim", type=int, default=16, help="embedding dim")
parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout rate")
parser.add_argument("--feature-encoder-hidden-size", type=int, default=32, help="hidden size of the feature encoder")

# ---------- TCN Encoder ----------
parser.add_argument("--TCN-list-channels", type=list, default=[8, 8, 8, 8, 8, 8, 8, 8], help="nums of channels of layers in TCN")
parser.add_argument("--TCN-hidden-size", type=int, default=8, help="hidden size in TCN")
parser.add_argument("--TCN-kernel-size", type=int, default=3, help="kernel size of TCN")

# ---------- header ----------
parser.add_argument("--header-channels-list", type=list, default=[16, 8], help="channels (dim) of each layer in MLP header")




# ========== Ablation Study ==========
parser.add_argument("--is-sequence-only", type=bool, default=False, help="Only use sequence")
parser.add_argument("--is-features-only", type=bool, default=False, help="Only use features")
parser.add_argument("--emotional-signals", type=str, default="hap-sad-rs", help="the signals input")
parser.add_argument("--is-use-type-emb", type=bool, default=True, help="if use type emb or not")
parser.add_argument("--is-combine", type=bool, default=False, help="if combine or not")


# =========== Comparison ==========
parser.add_argument("--model-name", type=str, default="ours", help="model name")

args = parser.parse_args()




def config_setup():
    args.logs_path = os.path.join("logs", args.experiment_name)
    args.logs_file = os.path.join(args.logs_path, ".log")
    args.csv_path = os.path.join(args.logs_path, "csv")
    args.model_path = os.path.join(args.logs_path, "model")
    args.config_file = os.path.join(args.logs_path, "config.json")

    with open(args.channels_name_file, "r") as channels_name_file:
        channels_name_dict = json.load(channels_name_file)
    args.data_channels_list = channels_name_dict[args.data_channels]
    args.input_size = int(len(args.data_channels_list) / 3) if args.is_combine else len(args.data_channels_list)

    args.device = torch.device("cuda" if args.is_use_cuda else "cpu")

    args.run_time = str(now)
    if args.input_data_type_name == "float":
        args.input_data_type = torch.float
    else:
        raise RuntimeError(f"no data type names {args.input_data_type_name}")

config_setup()
