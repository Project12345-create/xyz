# 1. Generate all possible images for a 3x2 black and white camera
import numpy as np
import matplotlib.pyplot as plt
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i in range(64):
    bits = [(i >> j) & 1 for j in range(6)]
    image = np.array(bits).reshape(3, 2)
    ax = axes[i // 8, i % 8]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(str(i), fontsize=7)
    ax.axis("off")
plt.suptitle("All 64 Possible 3x2 Images")
plt.tight_layout()
plt.show()


# 2. Randomly generate a 3x3 binary image until a sensible image appears
import numpy as np
import matplotlib.pyplot as plt
while True:
    image = np.random.randint(0, 2, (3, 3))
    print("Generated Matrix:")
    print(image)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title("Random 3x3 Binary Image")
    plt.axis("off")
    plt.show()
    choice = input("Is this image sensible? (y/n): ")
    if choice.lower() == "y":
        print("Sensible image selected.")
        break


# 3. Generate five Tamil characters using 5x5 binary matrices
import numpy as np
import matplotlib.pyplot as plt
characters = {
    "அ": [[0,1,1,1,0],
          [1,0,0,0,1],
          [1,1,1,1,1],
          [1,0,0,0,1],
          [1,0,0,0,1]],

    "ஆ": [[1,1,1,1,0],
          [1,0,0,0,1],
          [1,1,1,1,1],
          [1,0,0,0,1],
          [1,1,1,1,0]],

    "இ": [[1,1,1,1,1],
          [0,0,1,0,0],
          [0,0,1,0,0],
          [1,0,1,0,1],
          [0,1,1,1,0]],

    "ஈ": [[1,1,1,1,1],
          [0,0,1,0,0],
          [1,1,1,0,0],
          [1,0,0,0,1],
          [1,1,1,1,0]],

    "உ": [[1,0,0,0,1],
          [1,0,0,0,1],
          [1,0,0,0,1],
          [0,1,0,1,0],
          [0,0,1,0,0]]
}
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, (letter, matrix) in zip(axes, characters.items()):
    ax.imshow(matrix, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(letter)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(True)
plt.tight_layout()
plt.show()


# 4. Calculate coin area percentage at 10 distances and plot the graph
import cv2
import numpy as np
import matplotlib.pyplot as plt
distances = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
percentages = []
for i in range(1, 11):
    image = cv2.imread(f"coin_images/{i}.jpg")
    if image is None:
        print("Image not found:", i)
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Coin not detected in image", i)
        continue
    coin = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(coin)
    total_area = image.shape[0] * image.shape[1]
    percentage = (area / total_area) * 100
    percentages.append(percentage)
    print(f"Image {i}: Coin area = {percentage:.2f}%")
plt.plot(distances[:len(percentages)], percentages, "o-")
plt.xlabel("Camera Distance (cm)")
plt.ylabel("Coin Area (%)")
plt.title("Distance vs Percentage Coin Area")
plt.grid(True)
plt.show()


# 5. Read an image matrix and extract its 8 and 16 neighbourhoods
import cv2
import numpy as np
image = cv2.imread("images/image.jpg", 0)
if image is None:
    raise FileNotFoundError("Image not found")
row = int(input("Enter pixel row: "))
col = int(input("Enter pixel column: "))
if row < 2 or col < 2 or row >= image.shape[0] - 2 or col >= image.shape[1] - 2:
    raise ValueError("Choose a pixel at least 2 pixels away from the border")
print("Pixel value:", image[row, col])
# 8-neighbourhood: 3x3 region around the pixel
eight = image[row-1:row+2, col-1:col+2]
print("8-neighbourhood:")
print(eight)
# 16-neighbourhood: outer ring of a 5x5 region
region = image[row-2:row+3, col-2:col+3]
sixteen = region.copy()
sixteen[1:4, 1:4] = 0
print("16-neighbourhood (outer ring of 5x5 region):")
print(sixteen)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 6. Generate the negative image of a square image
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("images/square.jpg")
if image is None:
    raise FileNotFoundError("Image not found")
negative = 255 - image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative Image")
plt.axis("off")
plt.show()


# 7. Detect parallel lines captured at different perspective angles
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("images/lines.jpg")
if image is None:
    raise FileNotFoundError("Image not found")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, 3.14159 / 180,threshold=50, minLineLength=30, maxLineGap=10)
result = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines")
plt.axis("off")
plt.show()


# 8. Apply power-law transformations to a selfie image
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("images/selfie.jpg", 0)
if image is None:
    raise FileNotFoundError("Selfie not found")
gamma_values = [0.4, 0.7, 1.0, 1.5, 2.0]
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")
for i, gamma in enumerate(gamma_values, 2):
    normalized = image / 255.0
    transformed = np.uint8(255 * (normalized ** gamma))
    plt.subplot(2, 3, i)
    plt.imshow(transformed, cmap="gray")
    plt.title(f"Gamma = {gamma}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# 9. Apply a convolution kernel to an image containing equal-sized circles
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("images/circles.jpg", 0)
if image is None:
    raise FileNotFoundError("Circle image not found")
# Laplacian kernel for detecting edges
kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]], dtype=np.float32)
output = cv2.filter2D(image, -1, kernel)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray")
plt.title("Convolution Output")
plt.axis("off")
plt.show()


# 10. Add Gaussian noise and recover the image using a 5x5 averaging filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("images/selfie.jpg", 0)
if image is None:
    raise FileNotFoundError("Selfie not found")
original = image.astype(np.float32)
# Generate Gaussian noise with mean 110 and standard deviation 5
noise = np.random.normal(110, 5, original.shape)
noisy = np.clip(original + noise, 0, 255).astype(np.uint8)
# Apply a 5x5 averaging filter
kernel = np.ones((5, 5), np.float32) / 25
recovered = cv2.filter2D(noisy, -1, kernel)
# Calculate Mean Squared Error
mse = np.mean((original - recovered.astype(np.float32)) ** 2)
print("MSE between original and recovered image:", mse)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(recovered, cmap="gray")
plt.title("Recovered Image")
plt.axis("off")
plt.tight_layout()
plt.show()


# 11. Apply rotation, scaling and shearing using explicit matrices
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("images/selfie.jpg", 0)
if image is None:
    raise FileNotFoundError("Selfie not found")
height, width = image.shape
center_x = (width - 1) / 2
center_y = (height - 1) / 2
def transform_image(image, matrix):
    h, w = image.shape
    # Translate the origin to the image center
    to_center = np.array([[1, 0, -center_x],[0, 1, -center_y],[0, 0, 1]], dtype=float)
    from_center = np.array([[1, 0, center_x],[0, 1, center_y],[0, 0, 1]], dtype=float)
    transform = from_center @ matrix @ to_center
    inverse = np.linalg.inv(transform)
    output = np.zeros((h, w), dtype=np.uint8)
    # Map each output pixel back to the source image
    for y in range(h):
        for x in range(w):
            source = inverse @ np.array([x, y, 1.0])
            sx = int(round(source[0]))
            sy = int(round(source[1]))
            if 0 <= sx < w and 0 <= sy < h:
                output[y, x] = image[sy, sx]
    return output
angle = np.radians(30)
rotation = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0, 0, 1]
])
scaling = np.array([
    [1.5, 0, 0],
    [0, 1.5, 0],
    [0, 0, 1]
])
shearing = np.array([
    [1, 0.5, 0],
    [0, 1, 0],
    [0, 0, 1]
])
rotated = transform_image(image, rotation)
scaled = transform_image(image, scaling)
sheared = transform_image(image, shearing)
plt.figure(figsize=(12, 6))
for i, (img, title) in enumerate([
    (image, "Original"),
    (rotated, "Rotation 30 Degrees"),
    (scaled, "Scaling 1.5x"),
    (sheared, "Shearing")
], 1):
    plt.subplot(2, 2, i)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()


# 12. Track leaf apex coordinates and apply 1D FFT and 2D FFT
import cv2
import numpy as np
import matplotlib.pyplot as plt
video = cv2.VideoCapture("videos/leaf.mp4")
if not video.isOpened():
    raise FileNotFoundError("Video not found")
fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise ValueError("Invalid video frame rate")
coordinates = []
frame_number = 0
sample_interval = max(1, int(round(fps)))
def click_apex(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["point"] = (x, y)
while True:
    success, frame = video.read()
    if not success:
        break
    if frame_number % sample_interval == 0:
        data = {"point": None}
        cv2.namedWindow("Click Leaf Apex")
        cv2.setMouseCallback("Click Leaf Apex", click_apex, data)
        while data["point"] is None:
            display = frame.copy()
            cv2.putText(display, "Click the leaf apex",(10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.imshow("Click Leaf Apex", display)
            if cv2.waitKey(20) == 27:
                video.release()
                cv2.destroyAllWindows()
                raise SystemExit("Tracking stopped")
        x, y = data["point"]
        coordinates.append([y, x])  # row, column
    frame_number += 1
video.release()
cv2.destroyAllWindows()
coordinates = np.array(coordinates, dtype=float)
if len(coordinates) < 2:
    raise ValueError("At least two coordinate samples are required")
rows = coordinates[:, 0]
cols = coordinates[:, 1]
# Remove mean to emphasize coordinate variations
row_signal = rows - np.mean(rows)
col_signal = cols - np.mean(cols)
# Apply 1D FFT separately
row_fft = np.fft.fft(row_signal)
col_fft = np.fft.fft(col_signal)
# Apply 2D FFT to the two coordinate signals arranged as a matrix
signal_2d = np.vstack((row_signal, col_signal))
fft_2d = np.fft.fft2(signal_2d)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(rows)
plt.title("Row Coordinate vs Sample Number")
plt.subplot(2, 2, 2)
plt.plot(cols)
plt.title("Column Coordinate vs Sample Number")
plt.subplot(2, 2, 3)
plt.plot(np.abs(row_fft))
plt.title("1D FFT of Row Signal")
plt.subplot(2, 2, 4)
plt.plot(np.abs(col_fft))
plt.title("1D FFT of Column Signal")
plt.tight_layout()
plt.show()
plt.imshow(np.log1p(np.abs(np.fft.fftshift(fft_2d))), cmap="gray")
plt.title("2D FFT Magnitude Spectrum")
plt.colorbar()
plt.show()
print("Number of coordinate samples:", len(coordinates))
print("First five (row, column) coordinates:")
print(coordinates[:5])


# 13. Compare string art image histograms and apply histogram equalization
import cv2
import matplotlib.pyplot as plt
original = cv2.imread("images/original.jpg", 0)
string_art = cv2.imread("images/string_art.jpg", 0)
if original is None or string_art is None:
    raise FileNotFoundError("Check the original and string art image paths")
# Apply histogram equalization
equalized = cv2.equalizeHist(original)
# Save the equalized image for uploading to the website
cv2.imwrite("images/equalized.jpg", equalized)
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(original, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(string_art, cmap="gray")
plt.title("String Art Image")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(equalized, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.hist(original.ravel(), 256, [0, 256])
plt.title("Original Histogram")
plt.subplot(2, 3, 5)
plt.hist(string_art.ravel(), 256, [0, 256])
plt.title("String Art Histogram")
plt.subplot(2, 3, 6)
plt.hist(equalized.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")
plt.tight_layout()
plt.show()
