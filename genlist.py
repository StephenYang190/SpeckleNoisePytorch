import os


# files = os.listdir('../DataRoot/TEST/sythetics/a005')
# with open('../DataRoot/vail.lst', 'w') as f:
#     for file in files:
#         if '.png' in file:
#             # noise path
#             f.write('TEST/sythetics/a005/' + file + ' ')
#             # gt path
#             f.write('TEST/sythetics/gt/' + file)
#             f.write('\n')
files = os.listdir('/home/tongda/data/speckle/TEST/sythetics/a001')
with open('/home/tongda/data/speckle/TEST/sythetics/a001.lst', 'w') as f1:
    with open('/home/tongda/data/speckle/TEST/sythetics/a005.lst', 'w') as f2:
        with open('/home/tongda/data/speckle/TEST/sythetics/a010.lst', 'w') as f3:
            for file in files:
                if '.png' in file:
                    # noise path
                    f1.write('a001/' + file + '\n')
                    f2.write('a005/' + file + '\n')
                    f3.write('a010/' + file + '\n')
