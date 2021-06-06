import csv
import os


# Find *.txt
def txt_find(PATH):
    files = os.listdir(PATH)
    txts = []
    for i in files:
        if i[-3:] == 'txt':
            txts.append(i)
    return txts


# Convert other extension to jpg
def Extension_changer(PATH):
    files = os.listdir(PATH)
    for i in files:
        if i[-3:] != 'jpg' and i[-3:] != 'txt':
            if i[-4:] == 'jpeg' or i[-4:] == 'webp':
                count = 4
            else:
                count = 3
            os.rename(PATH + i, PATH + i[:-count] + 'jpg')


# Change image and txt. (class_name+num) (ex) Table_0.jpg
def name_changer(PATH, class_name, start_num):
    files = os.listdir(PATH)
    jpg_name = ''
    txt_name = ''
    count = start_num
    for i in range(len(files)):
        if files[i][-3:] == 'jpg':
            jpg_name = files[i]
        if files[i][-3:] == 'txt':
            txt_name = files[i]
        if jpg_name[:-3] == txt_name[:-3]:
            os.rename(PATH + jpg_name, PATH + class_name + str(count) + '.jpg')
            os.rename(PATH + txt_name, PATH + class_name + str(count) + '.txt')
            count += 1


# Change class (1 class -> multi class)
def class_chager(PATH, txts, class_num):
    print(txts)
    with open(PATH + txts, 'r') as csvfile:
        list_arr = []
        reader = csv.reader(csvfile, delimiter=' ')

        for row in reader:
            list_arr.append(row)

    for i in list_arr:
        i[0] = str(class_num)

    with open(PATH + txts, 'w') as csvfile:
        for i in list_arr:
            tmp = ''
            for j in i:
                tmp += j + ' '
            csvfile.write(tmp.strip() + '\n')


# ex) path = 'C:/Users/DH/Downloads/data/img/'  contains images, annotations
path = 'C:/Users/DH/Downloads/data/동훈_식당속책상/img/'


Extension_changer(PATH=path)
name_changer(PATH=path, class_name='Person_', start_num=0)
txts = txt_find(PATH=path)
for i in txts:
    class_chager(PATH=path, txts=i, class_num=1)  # 사람 -> 0, 책상 -> 1
