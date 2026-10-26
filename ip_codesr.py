import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import product
from PIL import Image

#1 all imgs 6 pixel
plt.figure(figsize=(10, 12))
for i, arr in enumerate(product([0, 1], repeat=6)):
    plt.subplot(8, 8, i+1)
    plt.imshow(np.array(arr).reshape(2, 3), cmap="gray")
    plt.axis("off")
plt.show()

#no library



#2 9 pixel random image
img = np.random.randint(0, 2, (3, 3))
plt.imshow(img, cmap="gray")
plt.title("3x3 Binary Image")
plt.axis("off")
plt.show()

#nolibrary
import random
while True:
    a = []
    for i in range(3):
        l = []
        for j in range(3):
            l.append(random.randint(0, 1))
        a.append(l)
    for row in a:
        print(row)
    break


#3tamil letters
chars = [
    np.array([
        [1,0,1,1,1],
        [1,0,1,0,1],
        [1,0,1,0,1],
        [1,0,1,0,1],
        [1,1,1,1,1]
        ]),
    np.array([[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]]),
    np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1]]),
    np.array([[1,1,1,1,1],[1,0,0,1,0],[1,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0]]),
    np.array([[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1],[0,1,1,1,1]])
]
fig, axis = plt.subplots(1, 5)
for axis, matrix in zip(axis, chars):
    axis.imshow(matrix, cmap="gray_r")
    axis.axis("off")
plt.show()

#no lib
chars = [
    [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,1,1,1,1]],
    [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
    [[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
    [[1,1,1,1,1],[1,0,0,1,0],[1,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0]],
    [[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1],[0,1,1,1,1]]
]
for matrix in chars:
    for row in matrix:
        print(" ".join("_" if x else " " for x in row))
    print()


#4 coin
distances = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
image_size = 100 * 100
coin_radius = 20
print("Distance  Coin Area  Percentage")
points = []
for d in distances:
    r = coin_radius / d * 10
    area = 3.14 * r * r
    percentage = round(area / image_size * 100 ,2)
    points.append((d, percentage))
    print(d, round(area, 2), round(percentage, 4))
print("\nGraph points (Distance, Percentage):")
for p in points:
    print(p)


#5 8 and 16 neighborhood
img = cv2.imread("image.jpg", 0)
img = cv2.resize(img, (100, 100))
r, c = np.random.randint(2, 98, 2)
print("Matrix:\n", img)
print("Pixel:", img[r, c])
n8 = img[r-1:r+2, c-1:c+2]
n16 = img[r-2:r+3, c-2:c+3]
print("8-neighbourhood:\n", n8)
print("16-neighbourhood:\n", n16)
plt.subplot(131), plt.imshow(img, cmap="gray"), plt.title("Image")
plt.subplot(132), plt.imshow(n8, cmap="gray"), plt.title("8-neighbourhood")
plt.subplot(133), plt.imshow(n16, cmap="gray"), plt.title("16-neighbourhood")
plt.show()


#no lib
a = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 1],
    [2, 3, 9, 5, 6],
    [7, 8, 1, 2, 3],
    [4, 5, 6, 7, 8]
]
r, c = 2, 2
print("Centre pixel:", a[r][c])
print("8-neighbourhood:")
for i in range(r-1, r+2):
    for j in range(c-1, c+2):
        if (i, j) != (r, c):
            print(a[i][j], end=" ")
    print()
print("16-neighbourhood (5x5 window excluding centre 3x3):")
for i in range(r-2, r+3):
    for j in range(c-2, c+3):
        if abs(i-r) > 1 or abs(j-c) > 1:
            print(a[i][j], end=" ")
    print()


#6 negative of image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
negative = 255 - img
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(negative, cmap="gray")
plt.title("Negative")
plt.axis("off")
plt.show()


#no lib
a = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
negative = [[1 - x for x in row] for row in a]
print("Original:")
for row in a:
    print(row)
print("Negative:")
for row in negative:
    print(row)


#7 parallel lines
print("Straight view:")
for i in range(7):
    print("|             |")
print("\nPerspective view:")
for i in range(7):
    spaces = i
    print(" " * spaces + "|" + " " * (12 - 2 * spaces) + "|")



#8 power law
img = cv2.cvtColor(cv2.imread('image.jpg'), cv2.COLOR_BGR2RGB)
for i, g in enumerate([0.4, 0.7, 1, 1.5, 2]):
    plt.subplot(2, 3, i+1)
    plt.imshow((255*(img/255)**g).astype(np.uint8))
    plt.title(f'Gamma={g}')
    plt.axis('off')

plt.show()

#nolib
import math
img = [
    [0, 50, 100, 150, 200, 255],
    [0, 50, 100, 150, 200, 255],
    [0, 50, 100, 150, 200, 255]
]
for g in [0.4, 0.7, 1, 1.5, 2]:
    print("Gamma =", g)
    for row in img:
        print([round(255 * (p/255)**g) for p in row])
    print()


#9 convolution output of sq/circle/triangle
image_array = cv2.imread('image.jpg', 0)
kernel = [[0,-1,0],[-1,4,-1],[0,-1,0]]
output = cv2.filter2D(image_array, cv2.CV_32F, np.array(kernel))
plt.subplot(121); plt.imshow(image_array, cmap='gray'); plt.axis('off')
plt.subplot(122); plt.imshow(output, cmap='gray'); plt.axis('off')
plt.show()


#no lib
a = [
    [0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0],
    [0, 255, 255, 255, 0],
    [0, 255, 255, 255, 0],
    [0, 0, 0, 0, 0]
]
k = [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]]
for i in range(1, 4):
    out = []
    for j in range(1, 4):
        s = 0
        for x in range(3):
            for y in range(3):
                s += a[i+x-1][j+y-1] * k[x][y]
        out.append(s)
    print(out)


#10 noise to image
img = np.array(Image.open('image.jpg').convert('L'), dtype=np.float32)
noisy = np.clip(img + np.random.normal(110, 5, img.shape), 0, 255)
recovered = cv2.blur(noisy, (5, 5))
print("Noisy MSE:", np.mean((img-noisy)**2))
print("Recovered MSE:", np.mean((img-recovered)**2))
for i, im in enumerate([img, noisy, recovered]):
    plt.subplot(1, 3, i+1)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
plt.show()

#no library
import random
a = [[100,120,140,160,180,200,220] for _ in range(7)]
n = len(a)
noisy = [[max(0, min(255, int(p + random.gauss(110, 5))))
          for p in row] for row in a]
rec = [r[:] for r in noisy]
for i in range(2, n-2):
    for j in range(2, n-2):
        rec[i][j] = sum(noisy[x][y]
                        for x in range(i-2, i+3)
                        for y in range(j-2, j+3)) / 25
mse = sum((a[i][j]-rec[i][j])**2
          for i in range(n) for j in range(n)) / (n*n)
print("Original:", a)
print("Noisy:", noisy)
print("Recovered:", rec)
print("MSE:", round(mse, 2))


#11 rotation, translation
img = [
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0]
]
def show(a):
    for row in a:
        print(" ".join(map(str, row)))
    print()
def transform(img, mode):
    h, w = len(img), len(img[0])
    points = []
    for y in range(h):
        for x in range(w):
            if img[y][x] == 1:
                X, Y = x - w // 2, y - h // 2
                if mode == "rotate":
                    nx, ny = -Y, X
                elif mode == "scale":
                    nx, ny = 2 * X, 2 * Y
                else:
                    nx, ny = X + Y, Y
                points.append((nx, ny))
    size = 11
    out = [[0] * size for _ in range(size)]
    for x, y in points:
        col = x + size // 2
        row = y + size // 2
        if 0 <= row < size and 0 <= col < size:
            out[row][col] = 1
    return out
print("Original:")
show(img)
print("Rotated:")
show(transform(img, "rotate"))
print("Scaled:")
show(transform(img, "scale"))
print("Sheared:")
show(transform(img, "shear"))


#13 string art generation 
a = [[0, 50, 50, 100],[100, 150, 150, 200],[200, 250, 250, 255]]
h = [0] * 256
for row in a:
    for p in row:
        h[p] += 1
total = len(a) * len(a[0])
b = []
cdf = 0
for i in range(256):
    cdf += h[i]
    b.append(round(cdf * 255 / total))
eq = [[b[p] for p in row] for row in a]
print("Original:", a)
print("Equalized:", eq)
