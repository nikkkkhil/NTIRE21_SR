# Modified from https://github.com/wonbeomjang/ESRGAN-pytorch/blob/master/datasets/preparation/download_image.py #

import zipfile
import tarfile
import urllib.request
import os
import shutil
from glob import glob
from tqdm import tqdm


def unzip_zip_file(zip_path, data_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()


def unzip_tar_file(zip_path, data_path):
    tar_ref = tarfile.open(zip_path, "r:")
    tar_ref.extractall(data_path)
    tar_ref.close()


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print("[!] download data file")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_dataset_train():

    DIV2K_HR_train = "https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_1X.zip"
    DIV2K_LR_train = "https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_4X.zip"

    if not os.path.exists('temp'):
        os.makedirs('temp')

    if not os.path.exists(os.path.join('./data/hr_train/')):
        os.makedirs(os.path.join('./data/hr_train/'))
    if not os.path.exists(os.path.join('./data/lr_train/')):
        os.makedirs(os.path.join('./data/lr_train/'))

    download_url(DIV2K_HR_train, os.path.join('temp', 'DIV2K-tr_1X.zip'))
    download_url(DIV2K_LR_train, os.path.join('temp', 'DIV2K-tr_4X.zip'))

    print('[!] Upzip zipfile')
    unzip_zip_file(os.path.join('temp', 'DIV2K-tr_1X.zip'), 'temp')
    unzip_zip_file(os.path.join('temp', 'DIV2K-tr_4X.zip'), 'temp')

    print('[!] Reformat DIV2K HR')
    image_path = glob('temp/DIV2K-tr_1X/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('./data/hr_train/', f'{index:04d}.png'))

    print('[!] Reformat DIV2K LR')
    image_path = glob('temp/DIV2K-tr_4X/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('./data/lr_train/', f'{index:04d}.png'))

    shutil.rmtree('temp')


def download_dataset_valid():

    DIV2K_HR_valid = "https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_1X.zip"
    DIV2K_LR_valid = "https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_4X.zip"

    if not os.path.exists('temp'):
        os.makedirs('temp')

    if not os.path.exists(os.path.join('./data/hr_valid/')):
        os.makedirs(os.path.join('./data/hr_valid/'))
    if not os.path.exists(os.path.join('./data/lr_valid/')):
        os.makedirs(os.path.join('./data/lr_valid/'))

    download_url(DIV2K_HR_valid, os.path.join('temp', 'DIV2K-va_1X.zip'))
    download_url(DIV2K_LR_valid, os.path.join('temp', 'DIV2K-va_4X.zip'))

    print('[!] Upzip zipfile')
    unzip_zip_file(os.path.join('temp', 'DIV2K-va_1X.zip'), 'temp')
    unzip_zip_file(os.path.join('temp', 'DIV2K-va_4X.zip'), 'temp')

    print('[!] Reformat DIV2K HR')
    image_path = glob('temp/DIV2K-va_1X/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('./data/hr_valid/', f'{index:04d}.png'))

    print('[!] Reformat DIV2K LR')
    image_path = glob('temp/DIV2K-va_4X/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('./data/lr_valid/', f'{index:04d}.png'))

    shutil.rmtree('temp')


if __name__ == "__main__":
    download_dataset_train()
    download_dataset_valid()