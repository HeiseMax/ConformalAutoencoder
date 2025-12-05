from json import decoder
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git


import torch
from torchvision import transforms
from matplotlib import pyplot as plt

from networks_medium import Encoder, Decoder
from Autoencoders import ConformalAutoencoder
from data import CelebA
from metrics import conformality_trace_loss, conformality_cosine_loss

from optuna import visualization as vis
import optuna
import optunahub

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = [transforms.CenterCrop((178, 178)), transforms.Resize((64, 64))]
    train_dataset = CelebA(root_dir="../", split="train", transform=transform, device=device, filter_categories=[(15, False), (20, True)])
    val_dataset = CelebA(root_dir="../", split="val", transform=transform, device=device, filter_categories=[(15, False), (20, True)])

    z_dim = 512
    in_ch = 3
    out_ch = 3
    base_ch = 32 #64
    gap_ch = 1


    def objective(trial):
        lambda_conf = trial.suggest_float("lambda_conf", 1e-6, 1.0, log=True)
        lambda_reg = trial.suggest_float("lambda_reg", 1e-7, 1e-1, log=True)
        conf_loss = trial.suggest_categorical("conformality_loss", ["trace", "cosine"]) #["trace","trace2", "cosine", "cosine2"]
        
        lambda_aug = 0.05
        lambda_conf_schedule = None
        lambda_reg_schedule = None
        num_samples_conf = 1
        num_samples_reg = 1

        epochs = 20
        batch_size = 512
        latent_dim = 512
        learning_rate = 0.001
        scheduler_kwargs={"step_size": 100, "gamma": 0.7}

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        encoder = Encoder(z_dim=z_dim, in_ch=in_ch, base_ch=base_ch, gap_ch=gap_ch)
        decoder = Decoder(z_dim=z_dim, out_ch=out_ch, base_ch=base_ch, gap_ch=gap_ch)

        encoder._build_fc_if_needed(16, 16)
        decoder._build_fc_if_needed(16, 16)

        conformal_autoencoder = ConformalAutoencoder(encoder, decoder, lambda_conf=lambda_conf, lambda_reg=lambda_reg, reg_in_loss=True, lambda_aug=lambda_aug).to(device)
        
        if conf_loss == "trace":
            conformal_autoencoder.conformality_loss = conformality_trace_loss
        elif conf_loss == "cosine":
            conformal_autoencoder.conformality_loss = conformality_cosine_loss

        #conformal_autoencoder.regularization_loss = lambda func, z: regularization(func, z, num_samples=num_samples_reg) #lambda func, z: 0.0


        optimizer, scheduler = conformal_autoencoder.train_model(train_dataloader, val_dataloader, has_label=True, epochs=epochs, batch_size=batch_size, learning_rate=1e-3, val_every=epochs, scheduler_kwargs={"step_size":20, "gamma":0.9})
        
        # stats, _ = evaluate_conformality(conformal_autoencoder, val_data[:1000])

        recon_error = conformal_autoencoder.val_metrics_list["reconstruction_loss"][-1]
        conf_error = conformal_autoencoder.val_metrics_list["conformal_loss"][-1]
        reg_error = conformal_autoencoder.val_metrics_list["regularization_loss"][-1]

        # create directory to save model
        if not os.path.exists(f"optuna_models/{study_name}"):
            os.makedirs(f"optuna_models/{study_name}")
        conformal_autoencoder.save_checkpoint(f"optuna_models/{study_name}/trial_{trial.number}.pth")

        return recon_error, conf_error, reg_error
    
    n_trials = 100
    study_name = "celeba_small_hyper2"
    storage = "sqlite:///optuna_celeba.db"

    # sampler = optuna.samplers.NSGAIISampler()
    module = optunahub.load_module(package="samplers/auto_sampler")
    sampler=module.AutoSampler()
    # search_space = {
    #     "lambda_conf": np.logspace(np.log10((0.01)), np.log10((100.0)), num=7),
    #     "lambda_reg": np.logspace(np.log10((0.0001)), np.log10((10.0)), num=7),
    #     "conformality_loss": ["trace", "trace2", "cosine"],
    # }
    # sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.create_study(directions=['minimize', 'minimize', 'minimize'], sampler=sampler, storage=storage, study_name=study_name, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    main()
