from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision
from torchinfo import summary
from matplotlib import pyplot as plt
import torch

from networks import Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

a = torch.randn(4, 3, 64, 64)
a = a.to(device)

latent_dim = 512
channels = 3
encoder = Encoder(gap_ch=1, z_dim=latent_dim, base_ch=32).to(device)
decoder = Decoder(gap_ch=1, z_dim=latent_dim, base_ch=32).to(device)

print(summary(encoder, input_size=(1, channels, 64, 64)))


from data import CelebA
CelebA_dataset = CelebA(split='train') #, filter_categories=[(15, False), (20, True)]


print(len(CelebA_dataset))


from torch.utils.data import DataLoader

dataloader = DataLoader(CelebA_dataset, batch_size=32, shuffle=False, num_workers=4)


for batch in dataloader:
    images, labels = batch
    break


plt.imshow(images[1].permute(1, 2, 0))
plt.show()



from data import parse_celeba_attr_file

filenames, classnames, labels = parse_celeba_attr_file('data/CelebA/list_attr_celeba.txt')
