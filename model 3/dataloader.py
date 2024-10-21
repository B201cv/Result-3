import os
import random
from collections import Counter
import cv2

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def trans_form_test(img):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((128,128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    img = transform(img)
    return img


def trans_form_train(img):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    img = transform(img)
    return img


class TrainData(Dataset):
    def __init__(self, root_dir1, root_dir2,root_dir3, root_dir4,root_dir5, training=True):
        self.palmvein_root_dir = root_dir1
        self.palmprint_root_dir = root_dir2
        self.knuckle_root_dir = root_dir3
        self.fingervein_root_dir = root_dir4
        self.print_root_dir = root_dir5
        self.person_path = os.listdir(self.print_root_dir)

    def __getitem__(self, idx):
        person_name = self.person_path[idx // 20]
        palmvein_path = os.listdir(os.path.join(self.palmvein_root_dir, person_name))
        palmprint_path = os.listdir(os.path.join(self.palmprint_root_dir, person_name))
        knuckle_path = os.listdir(os.path.join(self.knuckle_root_dir, person_name))
        fingervein_path = os.listdir(os.path.join(self.fingervein_root_dir, person_name))
        print_imgs_path = os.listdir(os.path.join(self.print_root_dir, person_name))
        length_imgs = len(print_imgs_path)
        sample1_index = random.sample(range(length_imgs), 1)
        sample2_index = random.sample(range(length_imgs), 1)
        sample3_index = random.sample(range(length_imgs), 1)
        sample4_index = random.sample(range(length_imgs), 1)
        sample5_index = random.sample(range(length_imgs), 1)

        palmvein_img_path = palmvein_path[sample1_index[0]]
        palmprint_img_path = palmprint_path[sample2_index[0]]
        knuckle_img_path = knuckle_path[sample3_index[0]]
        fingervein_img_path = fingervein_path[sample4_index[0]]
        print_img_path = print_imgs_path[sample5_index[0]]

        palmvein_img_item_path = os.path.join(self.palmvein_root_dir, person_name, palmvein_img_path)
        palmprint_img_item_path = os.path.join(self.palmprint_root_dir, person_name, palmprint_img_path)
        knuckle_img_item_path = os.path.join(self.knuckle_root_dir, person_name, knuckle_img_path)
        fingervein_img_item_path = os.path.join(self.fingervein_root_dir, person_name, fingervein_img_path)
        print_img_item_path = os.path.join(self.print_root_dir, person_name, print_img_path)

        palmvein_img = cv2.imread(palmvein_img_item_path)
        palmprint_img = cv2.imread(palmprint_img_item_path)
        knuckle_img = cv2.imread(knuckle_img_item_path)
        fingervein_img = cv2.imread(fingervein_img_item_path)
        print_img = cv2.imread(print_img_item_path)

        palmvein_img = torch.tensor(palmvein_img).to(torch.float).permute(2, 0, 1)
        palmprint_img = torch.tensor(palmprint_img).to(torch.float).permute(2, 0, 1)
        knuckle_img = torch.tensor(knuckle_img).to(torch.float).permute(2, 0, 1)
        fingervein_img  = torch.tensor(fingervein_img ).to(torch.float).permute(2, 0, 1)
        print_img = torch.tensor(print_img).to(torch.float).permute(2, 0, 1)

        palmvein_img = trans_form_train(palmvein_img)
        palmprint_img = trans_form_train(palmprint_img)
        knuckle_img = trans_form_train(knuckle_img)
        fingervein_img = trans_form_train(fingervein_img)
        print_img = trans_form_train(print_img)
        return palmvein_img,palmprint_img, print_img,knuckle_img,fingervein_img, person_name

    def __len__(self):
        return len(self.person_path) * 20


b = []
a = []
class TestData(Dataset):
    def __init__(self, root_dir1, root_dir2,root_dir3, root_dir4,root_dir5, img_num):
        self.palmvein_root_dir = root_dir1
        self.palmprint_root_dir = root_dir2
        self.knuckle_root_dir = root_dir3
        self.fingervein_root_dir = root_dir4
        self.print_root_dir = root_dir5
        self.person_path = os.listdir(self.print_root_dir)
        self.img_num = img_num

    def __getitem__(self, idx):
        person_name = self.person_path[idx // self.img_num]
        a.append(person_name)
        bb = Counter(a)
        b = bb[person_name] - 1
        palmvein_path = os.listdir(os.path.join(self.palmvein_root_dir, person_name))
        palmprint_path = os.listdir(os.path.join(self.palmprint_root_dir, person_name))
        knuckle_path = os.listdir(os.path.join(self.knuckle_root_dir, person_name))
        fingervein_path = os.listdir(os.path.join(self.fingervein_root_dir, person_name))
        print_imgs_path = os.listdir(os.path.join(self.print_root_dir, person_name))

        length1_imgs = len(print_imgs_path)
        if len(a) == len(print_imgs_path):
            a.clear()
        palmvein_img_path = palmvein_path[b]
        palmprint_img_path = palmprint_path[b]
        knuckle_img_path = knuckle_path[b]
        fingervein_img_path = fingervein_path[b]
        print_img_path = print_imgs_path[b]

        palmvein_img_item_path = os.path.join(self.palmvein_root_dir, person_name, palmvein_img_path)
        palmprint_img_item_path = os.path.join(self.palmprint_root_dir, person_name, palmprint_img_path)
        knuckle_img_item_path = os.path.join(self.knuckle_root_dir, person_name, knuckle_img_path)
        fingervein_img_item_path = os.path.join(self.fingervein_root_dir, person_name, fingervein_img_path)
        print_img_item_path = os.path.join(self.print_root_dir, person_name, print_img_path)

        palmvein_img = cv2.imread(palmvein_img_item_path)
        palmprint_img = cv2.imread(palmprint_img_item_path)
        knuckle_img = cv2.imread(knuckle_img_item_path)
        fingervein_img = cv2.imread(fingervein_img_item_path)
        print_img = cv2.imread(print_img_item_path)

        palmvein_img = torch.tensor(palmvein_img).to(torch.float).permute(2, 0, 1)
        palmprint_img = torch.tensor(palmprint_img).to(torch.float).permute(2, 0, 1)
        knuckle_img = torch.tensor(knuckle_img).to(torch.float).permute(2, 0, 1)
        fingervein_img = torch.tensor(fingervein_img).to(torch.float).permute(2, 0, 1)
        print_img = torch.tensor(print_img).to(torch.float).permute(2, 0, 1)

        palmvein_img = trans_form_train(palmvein_img)
        palmprint_img = trans_form_train(palmprint_img)
        knuckle_img = trans_form_train(knuckle_img)
        fingervein_img = trans_form_train(fingervein_img)
        print_img = trans_form_train(print_img)
        return palmvein_img, palmprint_img, print_img, knuckle_img, fingervein_img, person_name

    def __len__(self):
        return len(self.person_path) * self.img_num

