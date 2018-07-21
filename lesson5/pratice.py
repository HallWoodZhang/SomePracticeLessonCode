import matplotlib.pyplot as plt
import skimage
from skimage import io, data, color, img_as_float
from skimage import data_dir
from skimage.viewer import ImageViewer
import numpy as np

def ioshow(img = None):
    if img is not None:
        io.imshow(img)
        io.show()



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

    # img = data.astronaut()
    # img_gray = color.rgb2gray(img)
    # rows, cols = img_gray.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         if (img_gray[i, j] <= 0.5):
    #             img_gray[i, j] = 0

    #         else:
    #             img_gray[i, j] = 1
    # io.imshow(img_gray)
    # io.show()

    # img = data.astronaut()
    # img_idx_modified = img[:, :, 0] > 170
    # print(img_idx_modified)
    # img[img_idx_modified] = [0, 255, 0]
    # io.imshow(img)
    # io.show()


    # img = data.astronaut()
    # print(img.dtype.name)
    # print(img)

    # dst = img_as_float(img)
    # print(dst.dtype.name)
    # print(dst)
    # io.imshow(dst)
    # io.show()


    # img = data.camera()
    # img_gray = color.rgb2gray(img)
    # io.push(img)
    # io.push(img_gray)
    # print(io.image_stack)
    # io.imshow_collection(io.image_stack)
    # io.show()
    # # io.imshow(img_gray)
    # # io.show()


    # img = data.coffee()
    # hsv = color.convert_colorspace(img, 'RGB', 'HSV')
    # ioshow(img)
    # ioshow(hsv)

    # img = data.coffee()
    # gray = color.rgb2gray(img)
    
    # rows, cols = gray.shape
    # labels = np.zeros([rows, cols])

    # for i in range(rows):
    #     for j in range(cols):
    #         if gray[i, j] < 0.4:
    #             labels[i, j] = 0
    #         elif gray[i, j] < 0.75:
    #             labels[i, j] = 1
    #         else:
    #             labels[i, j] = 2
    
    # dst = color.label2rgb(labels)
    # ioshow(dst)


    # img = data.chelsea()
    # plt.figure(num="chelsea", figsize=(8,8))

    # plt.subplot(2, 2, 1)
    # plt.title("origin image")
    # plt.imshow(img)

    # plt.subplot(2, 2, 2)
    # plt.title("red channel")
    # plt.imshow(img[:,:,0], plt.cm.gray)
    # plt.axis('off')

    # plt.subplot(2, 2, 3)
    # plt.title("green channel")
    # plt.imshow(img[:,:,1], plt.cm.gray)
    # plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.title("blue channel")
    # plt.imshow(img[:,:,2], plt.cm.gray)
    # plt.axis('off')

    # plt.show()

    
    # img = data.coffee()
    # hsv = color.rgb2hsv(img)
    # fig, axes = plt.subplots(2, 2, figsize=(7,6))

    # ax0,ax1,ax2,ax3 = axes.ravel()

    # ax0.imshow(img)
    # ax0.set_title("ax0: original image")

    # ax1.imshow(hsv[:,:,0], cmap = plt.cm.gray)
    # ax1.set_title("ax1: H")

    # ax2.imshow(hsv[:,:,1], cmap = plt.cm.gray)
    # ax2.set_title("ax2: S")

    # ax3.imshow(hsv[:,:,2], cmap = plt.cm.gray)
    # ax3.set_title("ax3: V")

    # for ax in axes.ravel():
    #     ax.axis('off')

    # fig.tight_layout()
    # plt.show()


    # img = data.coins()
    # viewer = ImageViewer(img)
    # viewer.show()


    # def convert_gray(f):
    #     rgb = io.imread(f)
    #     return color.rgb2gray(rgb)
    
    # str = "../neu-dataset/elk/*.jpg"
    # coll = io.ImageCollection(str, load_func=lambda f: color.rgb2gray(io.imread(f)))
    # io.imshow(coll[1])
    # io.show()

    # class MP4Loader:
    #     video_file = './udk.mp4'

    #     def __call__(self, frame):
    #         return video_read(self.video_file, frame)

    # mp4_load = MP4Loader()
    # ic = io.ImageCollection([i for i in range(0, 1000, 10)], load_func=mp4_load)
    # print(ic)
    # print(skimage.io.concatenate_images(ic))


    



if __name__ == "__main__":
    main()