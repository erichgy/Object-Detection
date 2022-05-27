import copy
import time
import torch.autograd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import CarDataset
import custom_batch_sampler


def load_data():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_loaders = {}
    data_sizes = {}
    for phase in ['train', 'val']:
        data_set = CarDataset.CustomDataset(kind=phase, transform=transform)
        print(phase + "_positive:", data_set.get_postive_num())
        print(phase + "_negative:", data_set.get_negative_num())
        data_sampler = custom_batch_sampler.CustomBatchSampler(data_set.get_postive_num(), data_set.get_negative_num(),
                                                               4, 12)
        data_loader = DataLoader(data_set, batch_size=16, sampler=data_sampler, num_workers=4, drop_last=True)
        data_loaders[phase] = data_loader
        data_sizes[phase] = data_sampler.__len__()

    return data_loaders, data_sizes


def train_model(data_loaders, data_sizes, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()
    # model.state_dict() 网络中的参数
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            batch_num = 0
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # optimizer.zero_grad把梯度信息设置为0
                optimizer.zero_grad()
                with torch.autograd.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.max(outputs, 1)
                    print("outputs:", outputs)
                    print("labels:", labels)
                    print("preds:", preds)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if batch_num % 100 == 0:
                    print("第 %s 批！" % batch_num)
                    time_elapsed = time.time() - since
                    print('in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                batch_num += 1
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_statc_dict(best_model_weights)
    return model
