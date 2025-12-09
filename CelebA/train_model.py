import torch
from torchinfo import summary
from torchvision import transforms
from matplotlib import pyplot as plt
import argparse

from networks_medium import Encoder, Decoder
from Autoencoders import ConformalAutoencoder
from data import CelebA
from metrics import conformality_cosine_loss, conformality_trace_loss
from vgg_perceptual_loss import VGGPerceptualLoss

class Custom_loss(torch.nn.Module):
    def __init__(self, weights, device):
        super(Custom_loss, self).__init__()
        self.weights = weights
        self.perceptual_loss_fn = VGGPerceptualLoss().to(device)
        self.mse_loss_fn = torch.nn.MSELoss()
        self.l1_loss_fn = torch.nn.L1Loss()

    def forward(self, input, target):
        loss = self.weights[0] * self.mse_loss_fn(input, target) + \
               self.weights[1] * self.l1_loss_fn(input, target) + \
               self.weights[2] * self.perceptual_loss_fn(input, target)
        return loss


def main():
    hyperparams = parse_args()
    # device configuration
    if hyperparams.gpu is not None:
        device = torch.device(f"cuda:{hyperparams.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset
    transform = [transforms.CenterCrop((178, 178)), transforms.Resize((64, 64))]
    train_dataset = CelebA(root_dir="../", split="train", transform=transform, device=device, filter_categories=[(15, False), (20, True)])
    val_dataset = CelebA(root_dir="../", split="val", transform=transform, device=device, filter_categories=[(15, False), (20, True)])

    # hyperparameters
    epochs = hyperparams.steps
    bs = hyperparams.bs
    learning_rate = hyperparams.lr
    path = hyperparams.path
    val_every = hyperparams.val_every

    z_dim = hyperparams.z_dim
    in_ch = hyperparams.in_ch
    out_ch = hyperparams.in_ch
    base_ch = hyperparams.base_ch
    gap_ch = hyperparams.gap_ch

    lambda_conf = hyperparams.lambda_conf
    lambda_reg = hyperparams.lambda_reg
    lambda_aug = hyperparams.lambda_aug

    if hyperparams.recon_loss == 'mse':
        reconstruction_loss = torch.nn.MSELoss()
    elif hyperparams.recon_loss == 'l1':
        reconstruction_loss = torch.nn.L1Loss()
    elif hyperparams.recon_loss == 'perceptual':
        reconstruction_loss = VGGPerceptualLoss().to(device)
    elif hyperparams.recon_loss == 'custom':
        reconstruction_loss = Custom_loss(hyperparams.recon_loss_weights, device)

    if hyperparams.conf_loss == 'cosine':
        conformality_loss = conformality_cosine_loss
    elif hyperparams.conf_loss == 'trace':
        conformality_loss = conformality_trace_loss

    # model
    encoder = Encoder(z_dim=z_dim, in_ch=in_ch, base_ch=base_ch, gap_ch=gap_ch)
    decoder = Decoder(z_dim=z_dim, out_ch=out_ch, base_ch=base_ch, gap_ch=gap_ch)
    encoder._build_fc_if_needed(16, 16)
    decoder._build_fc_if_needed(16, 16)

    conformal_autoencoder = ConformalAutoencoder(encoder, decoder,
                                                lambda_conf=lambda_conf, 
                                                lambda_reg=lambda_reg,
                                                reg_in_loss=True,
                                                lambda_aug=lambda_aug
                                                ).to(device)

    conformal_autoencoder.conformality_loss = conformality_loss
    conformal_autoencoder.reconstruction_loss = reconstruction_loss

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True)


    optimizer, scheduler = conformal_autoencoder.train_model(train_loader,
                                                         val_loader,
                                                         has_label=True,
                                                         epochs=epochs,
                                                         batch_size=bs,
                                                         learning_rate=learning_rate,
                                                         val_every=val_every,
                                                         scheduler_kwargs={"step_size":20, "gamma":0.85},
                                                         checkpoint_path=f"{path}.pth"
                                                         )
    
    conformal_autoencoder.save_checkpoint(f"{path}.pth")

    torch.save(optimizer.state_dict(), f"{path}_optimizer.pth")
    torch.save(scheduler.state_dict(), f"{path}_scheduler.pth")
    

def parse_args():
    '''setup'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use. None means using all available GPUs.')
    # parser.add_argument('--manualSeed', default=48, type=int, help='random seed')
    parser.add_argument('--path', default='models/conformal_autoencoder_default', help='path to save model and log files')

    '''model'''
    parser.add_argument('--z_dim', type=int, default=1024, help='dimension of latent space')
    parser.add_argument('--base_ch', type=int, default=32, help='base channel number for autoencoder')
    parser.add_argument('--gap_ch', type=int, default=3, help='channel number for GAP layer in autoencoder')
    parser.add_argument('--in_ch', type=int, default=3, help='number of input image channels')

    '''training'''
    # parser.add_argument('--model', default='', help="path to model (to continue training)")

    parser.add_argument('--bs', type=int, default=512, help='input batch size')
    parser.add_argument('--steps', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for E, default=0.0002')

    parser.add_argument('--recon_loss', default='mse', choices=['mse', 'l1', 'perceptual', 'custom'], help='reconstruction loss type')
    parser.add_argument('--recon_loss_weights', type=float, nargs='*', default=[1.0, 1/20, 1/1000], help='weights for different reconstruction losses if using custom recon loss')
    parser.add_argument('--conf_loss', default='cosine', choices=['trace', 'cosine'], help='conformality loss type')
    parser.add_argument('--lambda_conf', type=float, default=0.01, help='weight for conformality loss')
    parser.add_argument('--lambda_reg', type=float, default=0.00056, help='weight for regularization loss')
    parser.add_argument('--lambda_aug', type=float, default=0.1, help='weight for augmentation loss')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    # parser.add_argument('--decay', type=float, default=0.0, help='weight decay for EBM')
    # parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    '''data'''
    # parser.add_argument('--dataset', type=str, choices=["mnist", "cifar10", "AFHQ"], default="mnist")
    # parser.add_argument('--classes', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], nargs="*")

    '''experiments'''
    parser.add_argument('--val_every', type=int, default=10, help='unit: epoch')
    # parser.add_argument('--valSamples', type=int, default=25, help="number of samples to create for tensorboard")
    # parser.add_argument('--valSteps', type=int, default=1000, help="number of integration-steps used for samples")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    main()
