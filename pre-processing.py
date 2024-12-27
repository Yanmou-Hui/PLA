import glob
import os
from tqdm import tqdm
import json


def save_split_txt(root):
    for root, dirs, files in os.walk(root):
        # if 'biggan' in root or 'cyclegan' in root or 'pggan' in root or 'stargan' in root or 'stylegan' in root or 'stylegan2' in root or 'ADM' in root or 'Midjourney' in root or 'stable_diffusion_v_1_4' in root or 'stable_diffusion_v_1_5' in root or 'DALLE2' in root or 'Glide' in root:
        if 'nature' in root:
            label = 1
            classname = 'real'
        else:
            label = 2
            classname = 'fake'
        for file in tqdm(files):
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') or file.endswith(
                    '.JPEG') or file.endswith('.JPG'):
                path = os.path.join(root, file)
                with open('../dataset/genimage/annotations/trainval.txt', 'a') as f:
                    f.write(f'{path}  {label}  {classname}\n')


def save_split_txt_LASTED(root):
    real_paintings = glob.glob(os.path.join(root, 'RealPainting_Danbooru', '*'))[0:80000]
    fake_paintings = glob.glob(os.path.join(root, 'SyntheticPainting_Lexica', '*'))[0:80000]
    real_photos = glob.glob(os.path.join(root, 'RealPhoto_LSUN', '*'))[0:80000]
    fake_photos = glob.glob(os.path.join(root, 'SyntheticPhoto_ProGAN', '*'))[0:80000]
    for path in tqdm(real_photos):
        with open('G:/mydataset/pic/LASTED_Trainset/annotations/trainval.txt', 'a') as f:
            f.write(f'{path}  1  real_photo\n')
    for path in tqdm(fake_photos):
        with open('G:/mydataset/pic/LASTED_Trainset/annotations/trainval.txt', 'a') as f:
            f.write(f'{path}  2  fake_photo\n')
    for path in tqdm(real_paintings):
        with open('G:/mydataset/pic/LASTED_Trainset/annotations/trainval.txt', 'a') as f:
            f.write(f'{path}  3  real_painting\n')
    for path in tqdm(fake_paintings):
        with open('G:/mydataset/pic/LASTED_Trainset/annotations/trainval.txt', 'a') as f:
            f.write(f'{path}  4  fake_painting\n')


def save_split_json(root):
    files_json = {}
    for root, dirs, files in os.walk(root):
        if ('biggan' in root or 'cyclegan' in root or 'pggan' in root or 'stargan' in root or 'stylegan' in root
                or 'stylegan2' in root or 'ADM' in root or 'Midjourney' in root
                or 'stable_diffusion_v_1_4' in root or 'stable_diffusion_v_1_5' in root or 'DALLE2' in root):
            if '1_fake' in root or '0_real' in root:
                dataset = root.split('\\')[1]
                classname = 'fake' if '1_fake' in root else 'real'
                label = 2 if '1_fake' in root else 1

                # 初始化数据集键
                if dataset not in files_json:
                    files_json[dataset] = []

                for file in tqdm(files):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        path = os.path.join(root, file)
                        files_json[dataset].append({'path': path, 'label': label, 'classname': classname})
    # JSON保存逻辑
    with open('G:/mydataset/pic/progan/annotations/test.json', 'w') as f:
        json.dump(files_json, f, indent=4)


def save_split_txt2(root_path, dataset_name):
    root_path = os.path.join(root_path, dataset_name)
    for root, dirs, files in os.walk(root_path):
        if '1_fake' in root:
            label = 14
            classname = f'{dataset_name}_fake'
        else:
            label = 13
            classname = f'{dataset_name}_real'
        for file in tqdm(files[:500]):
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') or file.endswith(
                    '.JPEG') or file.endswith('.JPG') or file.endswith('.PNG'):
                path = os.path.join(root, file)
                with open(f'G:/mydataset/pic/progan/annotations/testADM.txt', 'a') as f:
                    f.write(f'{path}  {label}  {classname}\n')


if __name__ == '__main__':
    save_split_txt('../dataset/genimage/stable_diffusion_v_1_4')
    # save_split_txt_LASTED('G:/mydataset/pic/LASTED_Trainset')
    # save_split_txt2('G:/mydataset/pic/progan/test', dataset_name='ADM')
