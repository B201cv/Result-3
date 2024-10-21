import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from LBCNN_train import BPVNet
from dataloader import TrainData, TestData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
    parser.add_argument("--num_class", type=int, default=532, help="epoch to start training from")
    parser.add_argument("--img_num", type=int, default=10, help="img_num of dataset")
    parser.add_argument("--dataset_name", type=str, default="QH", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
    parser.add_argument("--lambda1", type=float, default=5, help="identity loss weight")
    parser.add_argument("--lambda2", type=float, default=1, help="identity loss weight")
    opt = parser.parse_args()
    print(opt)

    def kd_loss(logits_student, logits_teacher, temperature):
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss_kd *= temperature ** 2
        return loss_kd

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = BPVNet(num_class=opt.num_class)
    if opt.epoch != 0:
        weights_path = "../Model_{}/path_{}/bestBPVNet.pth".format(opt.dataset_name, opt.dataset_name,
                                                                     opt.dataset_name, opt.epoch)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    loss_1 = nn.CrossEntropyLoss()
    loss_2 = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    data_train = TrainData("../../hand-multi-dataset/palmvein_train/",
                           "../../hand-multi-dataset/palmprint_train/",
                           "../../hand-multi-dataset/print_train/",
                           "../../hand-multi-dataset/knuckle_train/",
                           "../../hand-multi-dataset/fingervein_train/"
                           )

    data_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    train_steps = len(data_loader) * opt.batch_size

    data_test = TestData("../../hand-multi-dataset/palmvein_test/",
                         "../../hand-multi-dataset/palmprint_test/",
                         "../../hand-multi-dataset/print_test/",
                         "../../hand-multi-dataset/knuckle_test/",
                         "../../hand-multi-dataset/fingervein_test/", opt.img_num)

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    best_acc = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        save_path = '../Model_{}/path_{}/{}_{}_{}Net.pth'.format(opt.dataset_name, opt.dataset_name, opt.dataset_name,
                                                                 epoch + 1, 'BPV')
        os.makedirs('../Model_{}/path_{}'.format(opt.dataset_name, opt.dataset_name), exist_ok=True)
        model.train()
        acc = 0.0
        running_loss = 0.0
        train_bar = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            img1, img2, img3, img4, img5, person_name = data
            optimizer.zero_grad()
            label = [int(_) - 1 for _ in person_name]
            label = torch.tensor(label).to(device)
            x1,x2 = model(img1.to(device),
                        img2.to(device),
                        img3.to(device),
                        img4.to(device),
                        img5.to(device))

            loss0_1 = loss_1(x1, label.to(device))
            loss0_2 = loss_1(x2, label.to(device))
            #
            loss1 = loss_2(x1, x2)

            loss0 = loss0_1 + loss0_2
            # #
            # loss2_1 = loss_1(x_1, label.to(device))
            # loss2_2 = loss_1(x_2, label.to(device))
            # loss2_3 = loss_1(x_3, label.to(device))
            # loss2_4 = loss_1(x_4, label.to(device))
            # loss2_5 = loss_1(x_5, label.to(device))
            # loss2 = loss2_1 + loss2_2 + loss2_3 + loss2_4 + loss2_5

            ## 结束

            loss = loss0 + opt.lambda1 * loss1
            predict = torch.max(x1, dim=1)[1]
            acc += torch.eq(predict, label.to(device)).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     opt.n_epochs,
                                                                     running_loss / (step + 1))
        accurate = acc / train_steps
        print('num:{},train_accuracy:{:.4f},acc:{}'.format(train_steps, accurate, acc))

        # test
        if (epoch + 1) >= 50:
            model.eval()
            acc = 0.0
            test_bar = tqdm(test_loader, file=sys.stdout)
            with torch.no_grad():
                for step, data in enumerate(test_bar):
                    img1, img2, img3, img4, img5, person_name = data
                    x1, x2 = model(img1.to(device),
                                   img2.to(device),
                                   img3.to(device),
                                   img4.to(device),
                                   img5.to(device))
                    predict = torch.max(x1, dim=1)[1]
                    label = [int(_) - 1 for _ in person_name]
                    label = torch.tensor(label).to(device)
                    acc += torch.eq(predict, label.to(device)).sum().item()
                    test_steps = len(test_loader) * opt.batch_size
                    accurate = acc / test_steps

            print("num:{}, test_accuracy:{:.4f},acc:{}".format(test_steps, accurate, acc))
            if best_acc < accurate:
                best_acc = accurate
                best_batch = epoch + 1
                save_path = '../Model_{}/path_{}/bestBPVNet.pth'.format(opt.dataset_name, opt.dataset_name)
                torch.save(model.state_dict(), save_path)

            print("best_acc = ", best_acc)
            print("best_batch=", best_batch)

        if (epoch + 1) % opt.checkpoint_interval == 0 and (epoch + 1) >= 90:
            torch.save(model.state_dict(), save_path)
        if (accurate == 1.00 and (epoch + 1) >= 96):
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
