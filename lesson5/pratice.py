from skimage import io, data, color
from skimage import data_dir
import numpy as np

def main():
    # img = io.imread('./elk.jpg', as_gray=True)
    # io.imshow(img)
    # io.show()

    # print(data_dir)
    # img = data.checkerboard()
    # io.imshow(img)
    # io.imsave("checkboarder_copy.jpg", img)

    # img = data.chelsea()
    # io.imshow(img[:, :])
    # io.show()
    # print(type(img))
    # print(img.shape)
    # # height
    # print(img.shape[0])
    # # width
    # print(img.shape[1])
    # # access r,g,b and so on
    # print(img.shape[2])
    # # pixel num
    # print(img.size)
    # # max pix num
    # print(img.max())
    # # min pix num
    # print(img.min())
    # # avg pix val
    # print(img.mean())
    # print()

    # img = data.astronaut()
    # rows, cols, dims = img.shape
    # for i in range(5000):
    #     x = np.random.randint(0, rows)
    #     y = np.random.randint(0, cols)
    #     img[x, y, :] = 255
    # io.imshow(img)
    # io.show()

    img = data.astronaut()
    img_gray = color.rgb2gray(img)
    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i, j] <= 0.5):
                img_gray[i, j] = 0

            else:
                img_gray[i, j] = 1
    io.imshow(img_gray)
    io.show()
    


if __name__ == "__main__":
    main()