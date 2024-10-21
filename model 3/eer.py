import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from thop import profile
# from LBCNN_P import BPVNet

import argparse

from CRRtest import TestData,ca

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="QH", help="name of the dataset")
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data = TestData('D:/Casia_2modality/print-test', 'D:/Casia_2modality/vein-test')
    batch_size = 4

    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    print("data_loader = ", test_loader)
    print("start test......")

    model = ca()

    # input1 = torch.randn(1, 3, 224, 224)  # (1, 3, 224, 224)分别表示输入的图片数量，图像通道处，图像高度和宽度
    # input2 = torch.randn(1, 3, 224, 224)  # (1, 3, 224, 224)分别表示输入的图片数量，图像通道处，图像高度和宽度
    #
    # ## 统计flops, params
    # flops, params = profile(model, inputs=(input1, input2))
    #
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')


    weights_path = "CAISA.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model = model.to(device)

    leinei = []
    quanbu = []
    m = 0
    all_time = 0
    for epoch in range(1):
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for data in test_bar:
                p_img, v_img, person_name = data
                torch.cuda.synchronize()
                time_start = time.time()
                outputs = model(p_img.to(device), v_img.to(device))
                torch.cuda.synchronize()
                time_end = time.time()
                time_sum = time_end - time_start
                m = m + 1
                if m > 1:
                    # print(m)
                    all_time = all_time + time_sum
                # print('time_sum = ', time_sum)
                # print()

                if m == 500:
                    avg_time = all_time / 499
                    print('avg_time = ', avg_time)
                person_labels = [int(_) - 1 for _ in person_name]
                person_labels = torch.tensor(person_labels)
                outputs = outputs.cpu().numpy()
                outputs = torch.tensor(outputs).to(device)
                outputs = torch.squeeze(outputs).to(device)
                o1 = 1-outputs.reshape(-1)
                o1 = o1.tolist()
                quanbu.extend(o1)

                person_labels = person_labels.cpu().numpy()
                y_label = torch.LongTensor([person_labels]).cuda()  # gather函数中的index参数类型必须为LongTensor
                ln = outputs.gather(1, y_label.view(-1, 1))
                ln = 1 - ln.reshape(-1)
                ln = ln.tolist()
                leinei.extend(ln)

        leijian = [y for y in quanbu if y not in leinei]
        t1 = sum(leinei)/len(leinei)
        t2 = sum(leijian)/len(leijian)

        FAR = []
        FRR = []
        yuu = []

        print(t1)
        print(t2)

        for yu in np.arange(0, 1.1, (t2-t1)/100):
            zhengque = []
            for j in leinei:
                if j >= yu:
                    zhengque.append(1)
            bb = Counter(zhengque)
            bo = bb[1]
            FRR.append(bo / len(leinei)*100)

            zhengque2 = []
            for j in leijian:
                if j <= yu:
                    zhengque2.append(1)
            bb = Counter(zhengque2)
            bo = bb[1]
            FAR.append(bo / len(leijian)*100)
            yuu.append(yu)

        # Interpolate the FAR and FRR curves
        f = interp1d(FAR, FRR, kind='linear')
        eer = brentq(lambda x: f(x) - x, 0, 1)
        eer_frr = f(eer)

        print(f'EER: {eer:.2f}, at threshold: {eer_frr:.2f}')
        print(FRR)
        print(FAR)
        # print(yuu)

        plt.figure()
        plt.plot( FAR,FRR, color='darkorange', lw=2, label=f'ROC curve ')
        plt.plot([0, 10], [0, 10], color='navy', lw=2, linestyle='--') #绘制对角线
        plt.ylim(0,10)
        plt.xlim(0, 10)
        plt.xlabel('FAR')
        plt.ylabel('FRR')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    main()


