from cgi import print_form
import math
import sys
import torch
import utils
import time
def get_correct_number(y_pred, y_test):
    y_pred = torch.round(y_pred)
    correct_num = (y_pred == y_test).sum().float()
    return correct_num


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    criterion_age = torch.nn.L1Loss()
    criterion_gender = torch.nn.BCELoss()
    
    model.train()
    print_freq = 20
    start_time = time.time()

    total_correct_num, total_age_loss = 0, 0.

    for i, (input, target_age, target_gender) in enumerate(data_loader):
        input = input.to(device)

        target_gender = target_gender.reshape(-1,1).type(torch.FloatTensor).to(device)
        target_age = target_age.reshape(-1,1).type(torch.FloatTensor).to(device)

        #print(type(target_gender))

        age, gender = model(input)
        loss_age = criterion_age(age, target_age)
        loss_gender = criterion_gender(gender, target_gender)

        total_loss = loss_age + loss_gender

        #print(age, target_age, loss_age)

        # back propagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # statistic 
        batch_correct_num = get_correct_number(gender, target_gender)
        gender_accuracy = batch_correct_num / target_gender.shape[0] * 100.0
        age_losses = loss_age.item()
        gender_CE_loss = loss_gender.item()

        total_correct_num += batch_correct_num / target_age.shape[0]
        total_age_loss += loss_age.item()

        if( i % print_freq == 0):
            print(f"Epoch:{epoch}| [{i}/{len(data_loader)}] | gender_acc:{gender_accuracy:.2f}, age_loss:{age_losses:.4f}, gender_CE_loss:{gender_CE_loss:.4f} ")
        #if(i > 300):
        #    print(age, target_age)


    avg_acc = total_correct_num / len(data_loader) * 100.0
    avg_loss = total_age_loss / len(data_loader)
    print(f"Elapse time: {time.time()-start_time:.2f} | train_acc:{avg_acc:.2f} | train_age_loss:{avg_loss:.4f}")
    return avg_acc, avg_loss

@torch.no_grad()
def evaluate(data_loader, model, device):
    
    criterion_age = torch.nn.L1Loss()
    criterion_gender = torch.nn.BCELoss()

    total_correct_num, total_age_loss = 0, 0.
    print_freq = 10
    start_time = time.time()

    model.eval()
    for i, (input, target_age, target_gender) in enumerate(data_loader):
        input = input.to(device)
        target_gender = target_gender.reshape(-1,1).type(torch.FloatTensor).to(device)
        target_age = target_age.reshape(-1,1).type(torch.FloatTensor).to(device)

        age, gender = model(input)

        # loss
        loss_age = criterion_age(age, target_age)
        loss_gender = criterion_gender(gender, target_gender)

        # accuracy 
        batch_correct_num = get_correct_number(gender, target_gender)
        gender_accuracy = batch_correct_num / target_gender.shape[0] * 100.0
        age_losses = loss_age.item() 
        gender_CE_loss = loss_gender.item() 

        if( i % print_freq == 0):
            print(f"Test | [{i}/{len(data_loader)}] | gender_acc:{gender_accuracy:.2f}, age_loss:{age_losses:.4f}, gender_CE_loss:{gender_CE_loss:.4f} ")

        total_correct_num += batch_correct_num / target_age.shape[0]
        total_age_loss += loss_age.item()

    avg_acc = total_correct_num / len(data_loader) * 100.0
    avg_loss = total_age_loss / len(data_loader)
    print(f"Elapse time: {time.time()-start_time:.2f} | val_acc:{avg_acc:.2f} | val_age_loss:{avg_loss:.4f}")

    return avg_acc, avg_loss