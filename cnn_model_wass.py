import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import TensorDataset, DataLoader

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 128
EPOCHS = 300

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.2) # a=0.2 for LeakyReLU
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif (classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1) and classname.find('Conditional') == -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class Discriminator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.label_projector = nn.Linear(6, 6 * 48 * 48)
        self.convstack=nn.Sequential(
            nn.Conv2d(7, 64, 4, 2, 1),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.linear_stack=nn.Sequential(
            nn.Linear(512 * 6 * 6, 1)
        )
    
    def forward(self, x, labels):
        labels = self.label_projector(labels).view(-1, 6, 48, 48)
        x = torch.cat([x, labels], 1)
        x = self.convstack(x)
        x = x.view((-1, 512 * 6 * 6))
        x = self.linear_stack(x)
        return x
    
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(100+6, 512 * 12 * 12),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            ConditionalBatchNorm2d(256, 6),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            ConditionalBatchNorm2d(128, 6),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            ConditionalBatchNorm2d(64, 6),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.finalblock = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, labels):
        noise = torch.randn(labels.size(0), 100, device=DEVICE)
        full = torch.cat((labels, noise), 1)
        x = full
        x = self.projector(x).view(-1, 512, 12, 12)
        for layer in self.block1:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, labels)
            else:
                x = layer(x)
        for layer in self.block2:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, labels)
            else:
                x = layer(x)
        for layer in self.block3:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, labels)
            else:
                x = layer(x)
        x = self.finalblock(x)
        return x

    def show_output(self, device=DEVICE):
        self.eval()

        with torch.no_grad():
            labels = torch.arange(0, 36, dtype=torch.long).to(device)
            labels = labels % 6
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=6).float()

            fake_images = self(labels_one_hot)

            fake_images = fake_images.reshape(-1, 48, 48)
            fake_images_denorm = (fake_images + 1) / 2.0

            grid_img = vutils.make_grid(fake_images_denorm.unsqueeze(1), nrow=6)

            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Generator Output Images")
            plt.imshow(grid_img.permute(1, 2, 0).cpu())
            plt.show()

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.num_features=num_features
        self.batch_norm = nn.BatchNorm2d(num_features, affine=False)
        self.scale = nn.Linear(condition_dim, num_features)
        self.shift = nn.Linear(condition_dim, num_features)

    def forward(self, x, condition):
        output = self.batch_norm(x)
        scale = self.scale(condition).view(-1, self.num_features, 1, 1)
        shift = self.shift(condition).view(-1, self.num_features, 1, 1)
        output = output * scale + shift
        return output

class Training():

    def __init__(self):
        self.discriminator = Discriminator().to(DEVICE)
        self.generator = Generator().to(DEVICE)
        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)
        self.discriminatorOptim = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0, 0.9))
        self.generatorOptim = torch.optim.Adam(self.generator.parameters(), lr=1e-5, betas=(0, 0.9))

        #lambda_lr = lambda epoch: 1.0 if epoch < EPOCHS // 2 else 1.0 - (epoch - EPOCHS // 2) / (EPOCHS // 2)
        #self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.generatorOptim, lr_lambda=lambda_lr)
        #self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.discriminatorOptim, lr_lambda=lambda_lr)

        X_train = np.load("train_data.npy")
        y_train = np.load("train_labels.npy")

        X_tensor = torch.Tensor(X_train) 
        y_tensor = torch.Tensor(y_train)

        dataset = TensorDataset(X_tensor, y_tensor)
        self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        self.train_loss_disc=[]
        self.train_loss_gen=[]

    def load_data(self, filepath):
        return torch.Tensor(np.load(filepath)).to(DEVICE)
    
    def compute_gradient_penalty(self, real_samples, fake_samples, labels):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(DEVICE)
        
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates, labels)
        
        fake = torch.ones(real_samples.size(0), 1).to(DEVICE)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_epoch(self):
        LAMBDA_GP = 10

        data_iter = iter(self.dataloader)
        steps_per_epoch = len(self.dataloader) // 5 

        for i in range(steps_per_epoch):
            for _ in range(10):
                try:
                    real_data, real_labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    real_data, real_labels = next(data_iter)

                real_data = real_data.to(DEVICE).reshape((-1, 1, 48, 48))
                real_labels = real_labels.to(DEVICE)

                self.discriminatorOptim.zero_grad()
                fake_images = self.generator(real_labels)

                real_validity = self.discriminator(real_data, real_labels)
                fake_validity = self.discriminator(fake_images.detach(), real_labels)
                
                gp = self.compute_gradient_penalty(real_data, fake_images.detach(), real_labels)
                
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gp
                d_loss.backward()
                self.discriminatorOptim.step()

            self.generatorOptim.zero_grad()
            
            gen_labels = real_labels
            fake_images = self.generator(gen_labels)
            fake_validity = self.discriminator(fake_images, gen_labels)
            
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            self.generatorOptim.step()

        print(f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
 
        self.train_loss_disc.append(d_loss.item())
        self.train_loss_gen.append(g_loss.item())

    def train_n_epochs(self, n):
        for _ in range(n):
            print(str(_+1)+" epoch")
            self.train_epoch()
            #self.scheduler_D.step()
            #self.scheduler_G.step()
            if (_+1) % 10 == 0:
                #self.generator.show_output()
                torch.save(self.discriminator.state_dict(), f"discriminator_epoch_{_+1}.pt")
                torch.save(self.generator.state_dict(), f"generator_epoch_{_+1}.pt")
        

if __name__=="__main__":
    train = Training()
    loading=1
    if loading==1:
        train.discriminator.load_state_dict(torch.load("./wass_models/discriminator_epoch_310.pt", weights_only=True))
        #train.discriminator.eval()
        train.generator.load_state_dict(torch.load("./wass_models/generator_epoch_310.pt", weights_only=True))
        #train.train_n_epochs(EPOCHS)
        #train.generator.eval()

    else:
        train.train_n_epochs(EPOCHS)
        np.save("train_loss_disc_real.npy", np.array(train.train_loss_disc))
        np.save("train_loss_gen.npy", np.array(train.train_loss_gen))
        torch.save(train.discriminator.state_dict(), "discriminator.pt")
        torch.save(train.generator.state_dict(), "generator.pt")
    uk=EPOCHS
    while(True):
        a=input()
        if int(a)==-1:
            break
        train.generator.show_output()