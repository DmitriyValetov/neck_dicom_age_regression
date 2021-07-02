import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def train_epoch(model, loader, optimizer, loss_fn, scheduler, device='cuda'):
    running_loss = 0
    number_of_samples = 0
    
    pbar = tqdm(total = len(loader), desc='Training', position=0, leave=True)
    for _, (images, targets) in enumerate(loader):
        
        images, targets = images.to(device), targets.to(device)
        model.train()
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() #*images.shape[0]
        number_of_samples += images.shape[0]
        pbar.update()
        
    pbar.close()
    return running_loss/number_of_samples


def test_model(model, testloader, loss_fn, verbose=False, device='cuda'):
    
    all_prediction = []
    all_targets = []
    pbar = tqdm(total = len(testloader), desc='Testing', position=0, leave=True)
    with torch.no_grad():
        for _, (images, targets) in enumerate(testloader):
            model.eval()
            all_prediction += model(images.to(device)).cpu().numpy().tolist()
            all_targets    += targets.cpu().numpy().tolist()
            pbar.update()
        
    pbar.close()
    return loss_fn(torch.tensor(all_prediction), torch.tensor(all_targets))/len(all_targets)


def train_test(
    model,
    optimizer,
    scheduler,
    trainloader,
    testloader,
    loss_fn,
    num_epochs=30, 
    verbose=False, 
    device='cuda',
):

    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, trainloader, optimizer, loss_fn, scheduler, device)
        train_losses.append(train_loss)

        test_loss = test_model(model, testloader, loss_fn, verbose, device)
        test_losses.append(test_loss)
        
        if verbose:
            tqdm.write('Epoch: '+ str(epoch) + ', Train loss: ' + str(train_loss) + ', Test loss: ' + str(test_loss))
    
    return train_losses, test_losses


