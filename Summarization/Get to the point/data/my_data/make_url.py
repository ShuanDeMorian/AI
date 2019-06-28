import os

dataPath="stories"

test=''
val=''
train = ''

def save_url(filename,data):
    with open(filename,'w',encoding='utf-8') as f:
        f.write(data)
        f.close()

for file in os.listdir(dataPath):
    test_file = '_JT0SmBkgPE' # 58개
    val_file = '1M-7mb7ddSw'  # 40개

    if test_file in file:
        test += '\n' + file[:-6]
    elif val_file in file:
        val += '\n' + file[:-6]
    else:
        train += '\n' + file[:-6]

save_url('all_test.txt',test[1:])
save_url('all_val.txt',val[1:])
save_url('all_train.txt',train[1:])
