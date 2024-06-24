from experiment import Experiment
import wandb
import utils
from config import args, config_setup
import datetime

def main():
    config_setup()
    utils.setup_random_seed()
    if args.is_use_wandb:
        wandb.login()
    utils.mkdir(args.csv_path)
    utils.mkdir(args.model_path)
    e = Experiment(args.k_folds, args.num_epochs, args.num_classes, args.batch_size)
    e.run()
    utils.save_config(args, args.config_file)
    
    
    
if __name__ == "__main__":
    main()
    