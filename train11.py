import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as vdatasets
import torch.optim as optim
import torchvision.models as vmodels
import copy
import torch.nn as nn
from PIL import Image as pil_image
import torch.nn.functional as F


DATASET_DIR = "data"
def create_dataset(dataset_dir):
    # raw image를 가공하여 모델에 넣을 수 있는 인풋으로 변환합니다.
    data_transforms = {
        'TRAIN': transforms.Compose([
            transforms.Resize((224, 224)),  # 1. 사이즈를 224, 224로 통일.
            transforms.RandomHorizontalFlip(),  # 좌우반전으로 데이터셋 2배 뻥튀기
            transforms.ToTensor(),  # 2. PIL이미지를 숫자 텐서로 변환.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 3. 노멀라이즈
        ]),
        'VAL': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 이미지 데이터셋의 형태로 트레이닝과 밸리데이션 데이터셋을 준비합니다.
    # 이 ImageFolder클래스에 폴더를 집어 넣으면, raw이미지를 읽어서 데이터셋을 만들어 주는데,
    # 이 때, 폴더명이 classname(Supervised Learning에서 Label)이 됩니다.
    image_datasets = {x: vdatasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x])
                      for x in ['TRAIN', 'VAL']}

    nb_classes = len(image_datasets['TRAIN'].classes)

    return image_datasets, nb_classes

def create_dataloaders(image_datasets, training_batch_size, validation_batch_size, isShuffle):
  dataloaders = {'TRAIN': torch.utils.data.DataLoader(image_datasets['TRAIN'], batch_size=training_batch_size, shuffle=isShuffle),
                 'VAL': torch.utils.data.DataLoader(image_datasets['VAL'], batch_size=validation_batch_size, shuffle=isShuffle)
                }
  return dataloaders

def prepare_loss_function_and_optimizer(lr, model):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return loss_function, optimizer

def train(model, optimizer, loss_function, data_iterator, epoch):
    print("training epoch {}".format(epoch))
    # model을 학습 모드로 바꿔줍니다.
    model.train()

    nb_corrects = 0
    nb_data = 0
    loss_list = []

    for ix, (inputs, targets) in enumerate(data_iterator):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # 모델에 inputs를 넣어 출력값 outputs를 얻습니다.
        outputs = model(inputs)

        # 출력값과 실제값의 오차를 계산합니다.
        loss = loss_function(outputs, targets)
        loss_list.append(loss.item())

        # 실제 맞춘 갯수와 전체 갯수를 업데이트합니다.
        nb_corrects += (outputs.argmax(1) == targets).sum().item()
        nb_data += len(targets)

        # optimizer를 먼저 깔끔하게 초기화합니다.
        optimizer.zero_grad()

        # loss를 역전파합니다.
        loss.backward()

        # optimizer를 사용해 모델의 파라미터를 업데이트합니다.
        optimizer.step()

        if ix % 100 == 0:
            print(">> [{}] | loss: {:.4f}".format(ix, loss))

    epoch_accuracy = nb_corrects / nb_data
    epoch_avg_loss = torch.tensor(loss_list).mean().item()

    print("[training {:03d}] avg_loss: {:.4f} | accuracy: {:.4f}".format(epoch, epoch_avg_loss, epoch_accuracy))

def evaluate(model, loss_function, data_iterator, epoch):
    print("evaluation epoch {}".format(epoch))
    model.eval()

    nb_corrects = 0
    nb_data = 0
    loss_list = []

    for inputs, targets in data_iterator:
        with torch.no_grad():
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # 모델에 inputs를 넣어 출력값 outputs를 얻습니다.
            outputs = model(inputs)

            # 출력값과 실제값의 오차를 계산합니다.
            loss = loss_function(outputs, targets)
            loss_list.append(loss.item())

            # 실제 맞춘 갯수와 전체 갯수를 업데이트합니다.
            nb_corrects += (outputs.argmax(1) == targets).sum().item()
            nb_data += len(targets)

    epoch_accuracy = nb_corrects / nb_data
    epoch_avg_loss = torch.tensor(loss_list).mean().item()

    print("[validation {:03d}] avg_loss: {:.4f} | accuracy: {:.4f}".format(epoch, epoch_avg_loss, epoch_accuracy))

def run_all(image_datasets, model, lr, num_epochs):
    dataloaders = create_dataloaders(image_datasets, 32, 32, True)

    # loss_function과 optimizer를 준비합니다.
    loss_function, optimizer = prepare_loss_function_and_optimizer(lr, model)

    for epoch in range(num_epochs):
        # 트레이닝
        train(model, optimizer, loss_function, dataloaders['TRAIN'], epoch)
        # 밸리데이션
        evaluate(model, loss_function, dataloaders['VAL'], epoch)

    return model

def inference(img_path, model):
    model.eval()
    img = pil_image.open(img_path)
    inputs = image_datasets['TRAIN'].transform(img).unsqueeze(0).to(DEVICE)

    outputs = model(inputs)

    probs, predicted_labels = torch.topk(F.softmax(outputs, dim=1)[0], 5)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
    for p, label in zip(probs, predicted_labels):
        print("Prediction: {} ({:.1f}%)".format(int2label[label.item()], p.item() * 100))

class Resnet_fc(nn.Module):
    def __init__(self, base_model, nb_classes, toFreeze=False):
        super(Resnet_fc, self).__init__()

        base_model_copy = copy.deepcopy(base_model)
        self.feature_extractor = nn.Sequential(*list(base_model_copy.children())[:-2])

        if toFreeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True

        self.gap = nn.AvgPool2d(7, 1)
        self.linear = nn.Linear(2048, nb_classes)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.linear(x)

        return x


image_datasets, nb_classes = create_dataset(DATASET_DIR)
dataloaders = create_dataloaders(image_datasets, 32, 32, True)

base_resnet = vmodels.resnet50(pretrained=True)
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

frozen_tl_resnet = Resnet_fc(base_resnet, nb_classes, toFreeze=False).to(DEVICE)
frozen_tl_resnet = run_all(image_datasets, frozen_tl_resnet, 0.00001, 3)

int2label = {v: k for k, v in image_datasets['TRAIN'].class_to_idx.items()}

outputs = inference("images/Abyssinian_16.jpg", frozen_tl_resnet)