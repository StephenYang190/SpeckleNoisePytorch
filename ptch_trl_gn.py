import os
import numpy as np
import cv2


def readImage(filename, high, weight):
    img = cv2.imread(filename)
    img = cv2.resize(img, (high, weight), interpolation=cv2.INTER_CUBIC)

    return img


def gen_patches(img, scales=[1], patch_size=70, stride=20):
    h, w, _ = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        n1 = int((h_scaled - patch_size) / stride)
        n2 = int((w_scaled - patch_size) / stride)
        for i2 in range(n2 - 1):
            for i1 in range(n1 - 1):
                x = img_scaled[i1 * stride:i1 * stride + patch_size,
                    i2 * stride:i2 * stride + patch_size]
                patches.append(x)
    return patches


def main():
    # parameter
    cleanPath = "./fulldata/gt/"
    noisePath = "./fulldata/noise/"
    train_lstPath = "train.lst"
    outRoot = "./DataRoot/"
    outClean = "GT/"
    outNoise = "NOISE/"

    high = 320
    weight = 320
    patch_size = 128

    print("Begin.")
    if not os.path.exists(outRoot + outClean):
        os.mkdir(outRoot + outClean)
    if not os.path.exists(outRoot + outNoise):
        os.mkdir((outRoot + outNoise))

    out = open(outRoot + train_lstPath, 'w')
    files = os.listdir(cleanPath)

    i = 0
    k = 0
    for index, file in enumerate(files):
        print(file)
        if file.endswith(".png"):
            k = k + 1
            cleanImg = readImage(cleanPath + file, high, weight)
            cleanP = gen_patches(cleanImg, patch_size=patch_size)

            noiseImg = readImage(noisePath + file, high, weight)
            noiseP = gen_patches(noiseImg, patch_size=patch_size)

            for clean, noise in zip(cleanP, noiseP):
                cv2.imwrite(outRoot + 'GT/gt_' + str(i) + '.png', clean)
                cv2.imwrite(outRoot + 'NOISE/noise_' + str(i) + '.png', noise)

                out.write('NOISE/noise_' + str(i) + '.png ' + 'GT/gt_' + str(i) + '.png' + '\n')
                i = i + 1
    print("Total:" + str(k))
    print("Finished.")


if __name__ == '__main__':
    main()
