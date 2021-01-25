import os


files = os.listdir('../DataRoot/TEST/sythetics/a005')
with open('../DataRoot/vail.lst', 'w') as f:
    for file in files:
        if '.png' in file:
            # noise path
            f.write('TEST/sythetics/a005/' + file + ' ')
            # gt path
            f.write('TEST/sythetics/gt/' + file)
            f.write('\n')