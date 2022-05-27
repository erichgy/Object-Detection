import torch

import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn

import dataprocess
import finetune_mode as finetune
import fileuntils as fu
import torchvision.models as models

if __name__ == '__main__':
    index_dir = "./data/car_images_indexs.csv"
    cars_index = dataprocess.read_car_index(index_dir)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_loaders, data_sizes = finetune.load_data()
    model = models.alexnet(pretrained=True)
    #print(model)
    data_loader = data_loaders['train']
    # print(data_loaders)
    inputs, targets = next(data_loader.__iter__())
    # print(inputs[0].size(), type(inputs[0]))
    trans = transforms.ToPILImage()
    # print(type(trans(inputs[0])))
    # print("targets:",targets)
    # print("input_shape:",inputs.shape)

    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    #print("二分类：", model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = finetune.train_model(data_loaders, data_sizes, model, criterion, optimizer, lr_scheduler,
                                      device=device, num_epochs=10)
    fu.Create_folder('./models')
    torch.save(best_model.state_dict(), 'models/alexnet_car.pth')
