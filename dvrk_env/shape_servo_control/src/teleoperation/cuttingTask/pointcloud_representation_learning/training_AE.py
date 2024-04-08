import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from architecture import AutoEncoder, AutoEncoder2

from dataset_loader import AEDataset
import os
from torch import cumsum
from emd import earth_mover_distance
import matplotlib.pyplot as plt
import argparse


tradeoff_constant = 0.195#0.2

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0   
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
        
        data = sample.to(device)
        # print(batch_idx)   
        optimizer.zero_grad()
        recon_batch = model(data)

        d = earth_mover_distance(data, recon_batch, 
        transpose=True)

        loss_1 = (torch.sum(d))*tradeoff_constant
        #loss_1 = (d[0] / 2 + d[1] * 2 + d[2] / 3)*35.5#33#35.5#37#34#36
        loss_2 = model.get_chamfer_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
        loss = loss_1 + loss_2

        # import pdb; pdb.set_trace()
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(f"loss 1 and 2: {loss_1.item()}, {loss_2.item()}; Ratio: {loss_2.item()/loss_1.item()}")
    print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss/num_batch)) 

    train_loss /= num_batch 
    print('Train set: Average loss: {:.10f}\n'.format(train_loss))
    return train_loss

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    num_batch = 0  
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            num_batch += 1
            data = sample.to(device)
            recon_batch = model(data)

            d = earth_mover_distance(data, recon_batch, transpose=True)
            #loss_1 = (d[0] / 2 + d[1] * 2 + d[2] / 3)*20
            loss_1 = (torch.sum(d))*tradeoff_constant
            loss_2 = model.get_chamfer_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
            test_loss += loss_1.item() + loss_2.item()


    test_loss /= num_batch

    print('Test set: Average loss: {:.10f}\n'.format(test_loss))
    return test_loss

def plot_curves(xs, ys_1, ys_2, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title="train test losses 2 ball bonus", path="./figures"):
    fig, ax = plt.subplots()
    ax.plot(xs, ys_1, label=label_1)
    ax.plot(xs, ys_2, label=label_2)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    torch.manual_seed(2021)
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--train_path', type=str, help="train data path")
    parser.add_argument('--test_path', type=str, help="test data path")
    parser.add_argument('--weight_path', type=str, help="where to save the weights")
    parser.add_argument('--tradeoff_constant', default=0.195, type=float, help="a constant that balances the two losses")
    parser.add_argument('--train_len', default=71000, type=int, help="size of the desired training subset")
    
    args = parser.parse_args()
    tradeoff_constant = args.tradeoff_constant
    train_len = args.train_len

    train_path = args.train_path
    test_path = args.test_path
    train_dataset = AEDataset(train_path)
    test_dataset = AEDataset(test_path)
    train_dataset = torch.utils.data.Subset(train_dataset, range(0, train_len))
    
    batch_size= 300 #5000 #100 #512
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    weight_path = args.weight_path
    os.makedirs(weight_path, exist_ok=True)
    
    print(f"* tradeoff_constant: {tradeoff_constant}")
    print("train len: ", train_len)
    print("num training data: ", len(train_dataset))
    print("num test data: ", len(test_dataset))
    print("train data path:", train_path)
    print("test data path:", test_path)

    model = AutoEncoder(num_points=256, embedding_size=256).to(device)  # simple conv1D
    # model = AutoEncoder2(num_points=512*3).to(device)   # PointNet++
    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)
    train_losses = []
    test_losses = []
    epochs = 201 #1511 #151
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test_loss = test(model, device, test_loader, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if epoch % 50 == 0:
            xs = [i for i in range(epoch+1)]
            print("lenx: ", len(train_losses))

            plot_curves(xs, train_losses, test_losses, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title=f"train test losses cutting epoch {epoch}", path="./figures")      
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch_" + str(epoch)))


    xs = [i for i in range(epochs)]
    plot_curves(xs, train_losses, test_losses, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title=f"train test losses cutting epoch {epochs-1}", path="./figures")      
    torch.save(model.state_dict(), os.path.join(weight_path, "epoch_" + str(epochs-1)))
