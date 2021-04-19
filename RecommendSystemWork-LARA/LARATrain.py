from torch import optim
import os
import torchvision.utils as vutils
import numpy as np
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import LARADataSet

import warnings
warnings.filterwarnings("ignore")

# Arguments
BATCH_SIZE = 256
DIM = 5
LABEL_EMBED_SIZE = 5
ATTR_NUM = 20
HIDDIM=100
IMGS_TO_DISPLAY_PER_CLASS = 20
USER_ATTR_DIM=ATTR_NUM
LOAD_MODEL = False
EPOCHS=150

# Directories for storing model and output samples
model_path = os.path.join('./model', 'LARA')
if not os.path.exists(model_path):
    os.makedirs(model_path)



dataset = LARADataSet.LARA()
testdataset=LARADataSet.LARATestDataSet()
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                          drop_last=True)
test_data_loader=torch.utils.data.DataLoader(dataset=testdataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                          drop_last=True)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.AttrEmbedding=nn.Linear(ATTR_NUM+1,ATTR_NUM*DIM)
        self.GenHideOne=nn.Linear(ATTR_NUM*DIM,HIDDIM)
        self.GenHideTwo=nn.Linear(HIDDIM,HIDDIM)
        self.GenHideThree = nn.Linear(HIDDIM , USER_ATTR_DIM)


    def forward(self,X,z_fake):
        X=X.float()
        z_fake=z_fake.float()
        X = torch.cat((z_fake, X), dim=1)
        X=X.view(BATCH_SIZE,-1)
        X=self.AttrEmbedding(X)
        X=self.GenHideOne(X)
        X=torch.tanh(X)
        X=self.GenHideTwo(X)
        X=torch.tanh(X)
        X=self.GenHideThree(X)
        X=torch.sigmoid(X)

        return X


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.AttrEmbedding = nn.Linear(ATTR_NUM*2, ATTR_NUM * DIM)
        self.DisHideOne = nn.Linear(ATTR_NUM * DIM, HIDDIM)
        self.DisHideTwo = nn.Linear(HIDDIM, HIDDIM)
        self.DisHideThree = nn.Linear(HIDDIM, 1)


    def forward(self, X):
        X = self.AttrEmbedding(X)
        X = self.DisHideOne(X)
        X = torch.tanh(X)
        X = self.DisHideTwo(X)
        X = torch.tanh(X)
        X = self.DisHideThree(X)
        X = torch.sigmoid(X)

        return X


gen = Generator()
dis = Discriminator()

# Load previous model
if LOAD_MODEL:
    gen.load_state_dict(torch.load(os.path.join(model_path, 'gen.pkl')))
    dis.load_state_dict(torch.load(os.path.join(model_path, 'dis.pkl')))

# Model Summary
print("------------------Generator------------------")
print(gen.parameters())
print("------------------Discriminator------------------")
print(dis.parameters())

# Define Optimizers
g_opt = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=2e-5)
d_opt = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=2e-5)

# Loss functions
loss_fn = nn.BCELoss()


# Labels
real_label = torch.ones(BATCH_SIZE)
fake_label = torch.zeros(BATCH_SIZE)

# GPU Compatibility
is_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)
if is_cuda:
    gen, dis = gen.cuda(), dis.cuda()
    real_label, fake_label = real_label.cuda(), fake_label.cuda()


total_iters = 0
max_iter = len(data_loader)

# Training
for epoch in range(EPOCHS):
    gen.train()
    dis.train()

    for i, data in enumerate(data_loader):
        item_attr, positive_user_vec, negative_user_vec=data

        total_iters += 1

        # Loading data
        z_fake = torch.randn(BATCH_SIZE, 1)

        if is_cuda:
            item_attr=item_attr.float().cuda()
            positive_user_vec=positive_user_vec.float().cuda()
            negative_user_vec=negative_user_vec.float().cuda()

            z_fake = z_fake.float().cuda()

        # Generate fake data
        user_fake = gen(item_attr,z_fake)

        # Train Discriminator

        positive_user_item_vec=torch.cat((positive_user_vec,item_attr),dim=1)
        fake_user_item_vec=torch.cat((user_fake,item_attr),dim=1)
        negative_user_item_vec=torch.cat((negative_user_vec,item_attr),dim=1)


        fake_out = dis(fake_user_item_vec.detach())
        positive_user_item_out = dis(positive_user_item_vec.detach())
        negative_user_item_out=dis(negative_user_item_vec.detach())
        fake_out=fake_out.view(BATCH_SIZE)
        positive_user_item_out=positive_user_item_out.view(BATCH_SIZE)
        negative_user_item_out=negative_user_item_out.view(BATCH_SIZE)

        fake_loss=loss_fn(fake_out, fake_label)
        positive_loss=loss_fn(positive_user_item_out, real_label.view(BATCH_SIZE))
        negative_loss=loss_fn(negative_user_item_out, fake_label.view(BATCH_SIZE))


        d_loss = (fake_loss +positive_loss+negative_loss)/3

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # Train Generator
        fake_out = dis(fake_user_item_vec)
        g_loss = loss_fn(fake_out, real_label)

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        if i % 50 == 0:
            print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                  + "\td_loss:" + str(round(d_loss.item(), 4))
                  + "\tg_loss:" + str(round(g_loss.item(), 4))
                  )
            for i, data in enumerate(test_data_loader):
                test_item_attr, test_positive_user_vec, test_negative_user_vec = data

                # Loading data
                z_fake = torch.randn(BATCH_SIZE, 1)

                if is_cuda:
                    test_item_attr = test_item_attr.float().cuda()
                    test_positive_user_vec = test_positive_user_vec.float().cuda()
                    test_negative_user_vec = test_negative_user_vec.float().cuda()

                    z_fake = z_fake.float().cuda()


                test_user_fake = gen(item_attr, z_fake)
                test_item_attr, test_positive_user_vec, test_negative_user_vec = data

                z_fake = torch.randn(BATCH_SIZE, 1)
                if is_cuda:
                    test_item_attr = test_item_attr.float().cuda()
                    test_positive_user_vec = test_positive_user_vec.float().cuda()
                    test_negative_user_vec = test_negative_user_vec.float().cuda()

                    z_fake = z_fake.float().cuda()

                # Generate fake data
                test_user_fake = gen(item_attr, z_fake)
                positive_result = torch.mul(test_user_fake,test_positive_user_vec)
                negative_result = torch.mul(test_user_fake,test_negative_user_vec)
                print("positive similarity:",torch.sum(positive_result/BATCH_SIZE).item())
                print("negative similarity:",torch.sum(negative_result/BATCH_SIZE).item())
                print()
                break


    if (epoch + 1) % 5 == 0:
        torch.save(gen.state_dict(), os.path.join(model_path, 'gen.pkl'))
        torch.save(dis.state_dict(), os.path.join(model_path, 'dis.pkl'))

        epoch=epoch+1








