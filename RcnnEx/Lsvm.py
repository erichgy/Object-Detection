import copy
import random
import time
import torch
import custom_batch_sampler as cbs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import custom_classifier_dataset
import custom_hard_negative_dataset as hn_dataset
import torchvision.models as models
import torch.nn as nn
import torch.autograd

batch_positive_size = 32
batch_negative_size = 96
batch_total = 128


def load_data():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    remain_negative_list = list()

    for phase in ['train', 'val']:
        data_set = custom_classifier_dataset.CustomClassifierDataset(phase, transform)
        if phase is 'train':
            positive_list = data_set.get_positive()
            negative_list = data_set.get_negative()

            init_negative_idxs = random.sample(range(len(negative_list)), len(positive_list))
            init_negative_list = [negative_list[idx] for idx in init_negative_idxs]
            remain_negative_list = [negative_list[idx] for idx in range(len(negative_list)) if
                                    idx not in init_negative_idxs]

            data_set.set_negative(init_negative_list)
            data_loaders['remain'] = remain_negative_list

        sampler = cbs.CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(),
                                         batch_positive=batch_positive_size, batch_negative=batch_negative_size)
        data_loader = DataLoader(data_set, batch_size=batch_total, sampler=sampler, num_workers=8, drop_last=True)

        data_loaders[phase] = data_loader
        data_sizes[phase] = len(sampler)

    return data_loaders, data_sizes


def hinge_loss(outputs, labels):
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)
    return loss


def add_hard_negative(hard_negative_list, negative_list, add_negative_list):
    for item in hard_negative_list:
        if len(add_negative_list) == 0:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))
        if list(item['rect']) not in add_negative_list:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))


def get_hard_negatives(preds, cache_dicts):
    #FP 误报，边界不存在，检测出边界  0 -> 1
    #TN 真阴，边界不存在，未检测出边界 0 -> 0
    fp_mask = preds == 1
    tn_mask = preds == 0
    fp_rects = cache_dicts['rect'][fp_mask].numpy()
    fp_image_idx = cache_dicts['image_id'][fp_mask].numpy()

    tn_rects = cache_dicts['rect'][tn_mask].numpy()
    tn_image_idx = cache_dicts['image_id'][tn_mask].numpy()

    hard_negative_list = [{'rect': fp_rects[idx], 'image_id': fp_image_idx[idx]} for idx in range(len(fp_rects))]
    easy_negative_list = [{'rect': tn_rects[idx], 'image_id': tn_image_idx[idx]} for idx in range(len(tn_rects))]

    return hard_negative_list, easy_negative_list


def save_model(model, param):
    pass


def train_mode(data_loaders, data_sizes, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase is 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            data_set = data_loaders[phase].dataset
            print('{} - positive_num: {} - negative_num:{} - data size: {}'.format(phase, data_set.get_positive_num,
                                                                                   data_set.get_negative_num,
                                                                                   data_sizes[phase]))

            for inputs, labels, cache_dicts in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.autograd.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = float(running_corrects) / data_sizes[phase]

            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        train_dataset = data_loaders['train'].dataset
        remain_negative_list = data_loaders['remain']
        images = train_dataset.get_images()
        transform = train_dataset.get_transform()

        with torch.set_grad_enabled(False):
            remain_dataset = hn_dataset.CustomHardNegativeDataset(remain_negative_list, images, transform=transform)
            remain_dataloader = DataLoader(remain_dataset, batch_size=batch_total, num_workers=8, drop_last=True)

            negative_list = train_dataset.get_negatives()
            add_negative_list = data_loaders.get('add_negative', [])

            running_corrects = 0
            for inputs, label, cache_dicts in remain_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                hard_negative_list, easy_negative_list = get_hard_negatives(preds.cpu().numpy(), cache_dicts)
                add_hard_negative(hard_negative_list, negative_list, add_negative_list)

            remain_acc = float(running_corrects) / len(remain_negative_list)
            print('remain negative size:{},acc:{:.4f}'.format(len(remain_negative_list), remain_acc))

            train_dataset.set_negative(negative_list)
            tmp_sampler = cbs.CustomBatchSampler(train_dataset.get_positive_num(), train_dataset.get_negative_num(),
                                                 batch_positive_size, batch_negative_size)
            data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_total, sampler=tmp_sampler,
                                               num_workers=9, drop_last=True)
            data_loaders['add_negative'] = add_negative_list

            data_sizes['train'] = len(tmp_sampler)

        save_model(model, 'models/Linear_svm_alexnet_car_%d.pth' % epoch)
    time_elapsed = time.time() - since

    print('Train complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    data_loaders, data_sizes = load_data()

    model_path = './models/alexnet_car.pth'
    model = models.alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(num_features, num_classes)
    model = model.to(device)
