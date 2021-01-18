import os


files = os.listdir('/home/yang/workspace/speckle/DataRoot/TEST/sythetics/a001')
with open('test.lst', 'w') as f:
    for file in files:
        if '.png' in file:
            f.write(file)
            f.write('\n')