import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

class My_Data(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.listdir(root_dir)

    def __getitem__(self, item):
        img_name = self.img_dir[item]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)
        return img

    def __len__(self):
        return len(self.img_dir)

batch_size = 100
gen_features = 64
dis_features = 64
gen_in_channel = 120
data_dir = 'dataset/size64'
dataset = My_Data(data_dir)
dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False)

class Generator (nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_in_channel,
                               out_channels=gen_features * 8,
                               kernel_size=4,
                               bias=False),
            nn.BatchNorm2d(gen_features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=gen_features * 8,
                               out_channels=gen_features * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(gen_features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=gen_features*4,
                               out_channels=gen_features*2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(gen_features*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=gen_features*2,
                               out_channels=gen_features,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(gen_features),
            nn.ReLU(True),

            nn.ConvTranspose2d(gen_features,3,4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, dis_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_features, 2*dis_features,4, 2, 1, bias=False),
            nn.BatchNorm2d(2*dis_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*dis_features, 4*dis_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*dis_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * dis_features, 8 * dis_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8 * dis_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * dis_features, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
image_size = 64
noise = torch.randn(image_size, gen_in_channel, 1, 1, device=device)
Gen = Generator()
Dis = Discriminator()
Gen.to(device)
Dis.to(device)
criterion = nn.BCELoss()
optimizerGen = optim.Adam(Gen.parameters(), lr=0.0002, betas=(0.5, 0.999),)
optimizerDis = optim.Adam(Dis.parameters(), lr=0.0002, betas=(0.5, 0.999),)
fake_flag = 0
real_flag = 1

img_list = []
G_loss, D_loss = [], []
iters = 0
num_epoch = 101
for epoch in range(num_epoch):
    for i, data in enumerate(dataloader):
        Dis.zero_grad()
        # print(data.shape)
        # real_image = data[0].to(device)
        # print(real_image.shape)
        real_image = data.to(device)
        b_size = real_image.size(0)
        label = torch.full((b_size,), real_flag, dtype=torch.float, device=device)
        output = Dis(real_image).view(-1)
        Dis_real_loss = criterion(output, label)
        Dis_real_loss.backward()
        d_real_mean = output.mean().item()

        label.fill_(fake_flag)
        noise = torch.randn(b_size, gen_in_channel, 1, 1, device=device)
        fake_image = Gen(noise)

        output = Dis(fake_image.detach()).view(-1)
        Dis_fake_loss = criterion(output, label)
        Dis_fake_loss.backward()
        d_fake_mean = output.mean().item()

        Dis_loss = Dis_fake_loss + Dis_real_loss

        optimizerDis.step()

        ## Gentraining

        Gen.zero_grad()
        label.fill_(real_flag)
        output = Dis(fake_image).view(-1)
        Gen_loss = criterion(output, label)
        Gen_loss.backward()
        g_gen_mean = Gen_loss.mean().item()

        optimizerGen.step()

        if(i%500==0):
            print(f"{epoch}Fake:{d_fake_mean}, True:{d_real_mean}, Gen Loss:{g_gen_mean}")
            test_image = Gen(noise).detach().cpu()
            plt.imshow(np.transpose(torchvision.utils.make_grid(test_image[:2],
                                                                padding=0,
                                                                normalize=True).cpu(), (1, 2, 0)))
            G_loss.append(Gen_loss.item())
            D_loss.append(Dis_loss.item())

torch.save(Gen, 'Gennetator_{}.pth'.format(num_epoch))
torch.save(Dis, 'Discriminator_{}.pth'.format(num_epoch))
print('Model saved')

plt.figure(11)
plt.plot(G_loss)
plt.plot(D_loss)
plt.show()

# model = torch.load('Gennetator_500.pth')
# image = model(noise).cpu()
# plt.imshow(np.transpose(torchvision.utils.make_grid(image[:100],
#                                                     padding=0,
#                                                     normalize=True).cpu(), (1, 2, 0)))
# plt.show()
