from PIL import Image
import numpy as np
import math
from scipy.interpolate import lagrange as lag
from numpy.polynomial.polynomial import Polynomial
from tqdm import trange, tqdm
import itertools
import random
import time



def read_image(path):
      img = Image.open(path).convert('L')
    img_array = np.asarray(img)
    return img_array


def pixel_groups(pixel_array, k, h):
    KH = []
    for i in range(0, pixel_array.shape[0], k):
        for j in range(0, pixel_array.shape[1], h):
            pixel = list(pixel_array[i:i+k, j:j+h])
            pixel = list(itertools.chain.from_iterable(pixel))
            KH.append(pixel)
    return KH

def change_int(x):
    if x > 250:
        x = 250
    return x

def polynomial(groups, k, h, n, T):
       gen_fimgs = []
    gen_fimgs_temp = []
    gen_fimgs_all = []
    for block in groups:
        for i in range(1, n+1):
            block = (np.array(block)).reshape(k, h)
            x = [list(block[:, j]) for j in range(h)] 
            for temp in x:
                temp = [change_int(i) for i in temp]
                base = np.array([i ** _ for _ in range(k)])
                base = int(np.matmul(base, np.array(temp)))
                base %= 251
                gen_fimgs.append(base)
            gen_fimgs_temp.append(gen_fimgs)
            gen_fimgs = []
        gen_fimgs_all.append(gen_fimgs_temp)
        gen_fimgs_temp = []
    # print(len(gen_fimgs_all[0][0]))
    gen_gimgs = []
    gen_gimgs_temp = []
    gen_gimgs_all = []
    for block in groups:
        for i in range(1, n+1):
            block = (np.array(block)).reshape(k, h)
            y = [list(block[j, :]) for j in range(k)]
            for temp in y:
                temp = [change_int(i) for i in temp]
                base = np.array([i ** _ for _ in range(h)])
                base = int(np.matmul(base, np.array(temp)))
                base %= 251
                gen_gimgs.append(base)
            gen_gimgs_temp.append(gen_gimgs)
            gen_gimgs = []
        gen_gimgs_all.append(gen_gimgs_temp)
        gen_gimgs_temp = []
    # print(len(gen_gimgs_all),len(gen_gimgs_all[0]))
    e = random.choice([i for i in range(n, 2*n)])
    gen_fgimgs = []
    gen_fgimgs_temp = []
    gen_fgimgs_all = []
    for block in groups:
        for i in range(1, n+1):
            block = (np.array(block)).reshape(k, h)
            y = [list(block[j, :]) for j in range(k)]
            for temp in y:
                temp = [change_int(i) for i in temp]
                base_temp = [i ** _ for _ in range(h)]
                for change in range(1, h-T+1):
                    change_temp = [_ for _ in range(h)]
                    base_temp[-change] = e**(change_temp[-change])
                base = np.array(base_temp)
                base = int(np.matmul(base, np.array(temp)))
                base %= 251
                gen_fgimgs.append(base)
            gen_fgimgs_temp.append(gen_fgimgs)
            gen_fgimgs = []
        gen_fgimgs_all.append(gen_fgimgs_temp)
        gen_fgimgs_temp = []
    return gen_fimgs_all, gen_gimgs_all, gen_fgimgs_all


def update(f_par, g_par, fg_par, k, h, n, T):
    if T == k:
        gen_imgs_all = []
        for i in range(1, n+1):
            gen_imgs_temp = []
            for par in f_par:
                y = np.array([i ** _ for _ in range(h)])
                pixel = int(np.matmul(par[i-1], y))
                pixel %= 251
                gen_imgs_temp.append(pixel)
            gen_imgs_all.append(gen_imgs_temp)
        par = f_par
    elif T == h:
        gen_imgs_all = []
        for i in range(1, n+1):
            gen_imgs_temp = []
            for par in g_par:
                y = np.array([i ** _ for _ in range(k)])
                pixel = int(np.matmul(par[i-1], y))
                pixel %= 251
                gen_imgs_temp.append(pixel)
            gen_imgs_all.append(gen_imgs_temp)
        par = g_par
    else:
        gen_imgs_all = []
        for i in range(1, n+1):
            gen_imgs_temp = []
            for par in fg_par:
                y = np.array([i ** _ for _ in range(k)])
                pixel = int(np.matmul(par[i-1], y))
                pixel %= 251
                gen_imgs_temp.append(pixel)
            gen_imgs_all.append(gen_imgs_temp)
        par = fg_par
    return gen_imgs_all,par


def lagrange(slave_img, k, h, n, M):
    x = np.array([i for i in range(1, M+1)])
    result = []
    for l in trange(len(slave_img)):
        y = np.array([i[0] for i in slave_img[l][:M]])
        poly = lag(x, y)
        temp = [int(poly(i))%251 for i in range(k*h)]
        result.append(temp)
    re_img = []
    for block in result:
        block = np.array(block)
        block = block.reshape(k, h)
        re_img.append(block)
    final = []
    for cnt in range(0,len(re_img),int(256/h)):
        for j in range(k):
            for block in re_img[cnt:cnt+int(256/h)]:
                final.append(block[j,:])
    final = list(itertools.chain.from_iterable(final))
    return final

def T_M_choose(k, h, n):
    T = k
    # T = h
    # T = random.choice([i for i in range(k,h+1)])#k<T<h
    M = T
    assert M >= T
    return T, M


def attack(slave_img, k, h, n, M,img_array):
    x = np.array([i for i in range(1, M+1)])
    collect_score = []
    collect_image = []
    for pixel in trange(256):
        result = []
        for l in range(len(slave_img)):
            y = np.array([i[0] for i in slave_img[l][:M]])
            y[-1] = pixel
            poly = lag(x, y)
            temp = [int(poly(i))%251 for i in range(k*h)]
            result.append(temp)
        re_img = []
        for block in result:
            block = np.array(block)
            block = block.reshape(k, h)
            re_img.append(block)
        final = []
        for cnt in range(0,len(re_img),int(256/h)):
            for j in range(k):
                for block in re_img[cnt:cnt+int(256/h)]:
                    final.append(block[j,:])
        final = list(itertools.chain.from_iterable(final))
        score = np.corrcoef(np.array(final).reshape(256, 256).astype(np.uint8), img_array)[0, 1]
        collect_score.append(score)
        collect_image.append(final)
    # print(max(collect_score))
    max_index = collect_score.index(max(collect_score))
    final = collect_image[max_index]
    return final

def main():
    factor = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    k = factor[1]
    h = factor[2]
    n = 9  
    path = "figures/Lina.bmp"
    img_array = read_image(path)
    KH = pixel_groups(img_array, k, h)
    T, M = T_M_choose(k, h, n)
    gen_fimgs_all, gen_gimgs_all, gen_fgimgs_all = polynomial(KH, k, h, n, T)
    gen_imgs_all,par = update(gen_fimgs_all, gen_gimgs_all,
                          gen_fgimgs_all, k, h, n, T)
    for i, img in enumerate(gen_imgs_all):
        Image.fromarray(np.array(img).reshape(int(256/h), int(256/k)).astype(np.uint8)).save(
            "shadows/temp_Lina_{}.bmp".format(i + 1))
    image = lagrange(par, k, h, n, M)
    Image.fromarray(np.array(image).reshape(256, 256).astype(np.uint8)).save(
        "shadows/re_Lina.bmp")
    start = time.time()
    image = attack(par, k, h, n, M,img_array)
    Image.fromarray(np.array(image).reshape(256, 256).astype(np.uint8)).save(
        "shadows/attack_Lina.bmp")
    end = time.time()
    elapsed = end - start
    print("The program took %d seconds to run." % elapsed)

if __name__ == '__main__':
    main()

