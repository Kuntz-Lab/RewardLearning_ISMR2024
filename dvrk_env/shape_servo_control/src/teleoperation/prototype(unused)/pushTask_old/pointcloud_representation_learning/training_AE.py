import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from architecture import AutoEncoder, AutoEncoder2

from dataset_loader import AEDataset
import os
from torch import cumsum
from emd import earth_mover_distance




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
        # loss = F.mse_loss(data, recon_batch)
        # loss = chamfer_distance(torch.swapaxes(data, 1, 2), torch.swapaxes(recon_batch, 1, 2))
        # loss = model.get_loss(data, recon_batch)
        # loss = model.get_chamfer_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))  # success
        # loss = F.mse_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
        # loss = F.mse_loss(data[:,0,:].squeeze(), recon_batch[:,0,:].squeeze()) + F.mse_loss(data[:,1,:].squeeze(), recon_batch[:,1,:].squeeze()) + F.mse_loss(data[:,2,:].squeeze(), recon_batch[:,2,:].squeeze())

        # loss = model.get_loss(data[:,:,0:256].permute(0,2,1), recon_batch[:,:,0:256].permute(0,2,1)) \
        #        + model.get_loss(data[:,:,256:512].permute(0,2,1), recon_batch[:,:,256:512].permute(0,2,1)) \
        #        + model.get_loss(data[:,:,512:768].permute(0,2,1), recon_batch[:,:,512:768].permute(0,2,1))

        # loss_1 = model.get_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
        # loss_2 = entropy(recon_batch[:,0,:].squeeze()) + entropy(recon_batch[:,1,:].squeeze()) + entropy(recon_batch[:,2,:].squeeze())
        # loss =  -loss_2 #loss_1 - loss_2

        # loss_1 = model.get_huber_loss(data, recon_batch)
        # loss_2 = model.get_chamfer_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
        # loss = loss_1 #+ loss_2 

        d = earth_mover_distance(data, recon_batch, 
        transpose=True)
        loss_1 = (d[0] / 2 + d[1] * 2 + d[2] / 3)*30.9#30
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

            # test_loss += F.mse_loss(data, recon_batch)
            # test_loss += model.get_loss(data, recon_batch).item()
            # test_loss += model.get_chamfer_loss(data.permute(0,2,1), recon_batch.permute(0,2,1)).item() # success 
            # test_loss += F.mse_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
            # test_loss += model.get_loss(data[:,:,0:256].permute(0,2,1), recon_batch[:,:,0:256].permute(0,2,1)) \
            #     + model.get_loss(data[:,:,256:512].permute(0,2,1), recon_batch[:,:,256:512].permute(0,2,1)) \
            #     + model.get_loss(data[:,:,512:768].permute(0,2,1), recon_batch[:,:,512:768].permute(0,2,1))

            # criterion = HLoss()
            # test_loss += model.get_loss(data.permute(0,2,1), recon_batch.permute(0,2,1)) \
            #         + criterion(data[:,0,:].squeeze())
            # loss_1 = model.get_huber_loss(data, recon_batch)
            # loss_1 = model.get_huber_loss(data, recon_batch)
            # loss_2 = model.get_chamfer_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
            # test_loss += loss_1 #+ loss_2 

            d = earth_mover_distance(data, recon_batch, transpose=True)
            loss_1 = (d[0] / 2 + d[1] * 2 + d[2] / 3)*20
            loss_2 = model.get_chamfer_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
            test_loss += loss_1 + loss_2 


    test_loss /= num_batch

    print('Test set: Average loss: {:.10f}\n'.format(test_loss))

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

    train_len = 40000#8000#10000
    test_len = 1000
    total_len = train_len + test_len

    # dataset_path = "/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_2/autoencoder/data_on_table_3_objects"
    dataset_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone_corrected/processed_data_train"
    dataset = AEDataset(dataset_path)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

    
    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    print("data path:", dataset.dataset_path)



    model = AutoEncoder(num_points=256, embedding_size=256).to(device)  # simple conv1D
    # model = AutoEncoder2(num_points=512*3).to(device)   # PointNet++
    model.apply(weights_init)

    # weight_path = "/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_2/autoencoder/weights"
    weight_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone_corrected/weights/weights_10000"
    # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch " + str(31))))
    # pretrained_path = "/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_2/autoencoder/weights_3_objects_EMD"
    # model.load_state_dict(torch.load(os.path.join(pretrained_path, "epoch " + str(60))))
    os.makedirs(weight_path, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(0, 151):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)
        
        if epoch % 1 == 0:            
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch)))

