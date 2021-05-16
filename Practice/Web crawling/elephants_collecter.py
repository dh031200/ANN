import os
import shutil
from google_images_download import google_images_download


def imageCrawling(keyword, dir):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": keyword,
                 "limit": 300,
                 "chromedriver": "/home/EIEN443/chromedriver",
                 "print_urls": False,
                 "no_directory": True,
                 "output_directory": dir}
    path = response.download(arguments)
    print(path)


# crawled path
african = './datasets/african/'
indian = './datasets/Indian/'

# African elephant
imageCrawling('forest elephant', african)
imageCrawling('African bush elephant', african)
imageCrawling('sub-Saharan Africa elephants', african)

# Indian elephant
imageCrawling('Asian elephants', indian)
imageCrawling('Indian elephants', indian)
imageCrawling('Elephas maximus', indian)

# Set path
train_path = './datasets/train/'
valid_path = './datasets/valid/'
test_path = './datasets/test/'


def mk_dir(path):
    os.mkdir(path)
    os.mkdir(path + 'Indian_elephant')
    os.mkdir(path + 'African_elephant')


mk_dir(train_path)
mk_dir(valid_path)
mk_dir(test_path)

# Split data
img_list = os.listdir(african)
for img in img_list[:500]:
    shutil.move(african + img, train_path + '/African_elephant')
for img in img_list[500:700]:
    shutil.move(african + img, valid_path + '/African_elephant')
for img in img_list[700:]:
    shutil.move(african + img, test_path + '/African_elephant')
print('African elephant dataset moved')
img_list = os.listdir('./datasets/Indian')
for img in img_list[:500]:
    shutil.move(indian + img, train_path + '/Indian_elephant')
for img in img_list[500:700]:
    shutil.move(indian + img, valid_path + '/Indian_elephant')
for img in img_list[700:]:
    shutil.move(indian + img, test_path + '/Indian_elephant')
print('Indian elephant dataset moved')
