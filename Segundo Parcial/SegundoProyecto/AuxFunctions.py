import cv2
import matplotlib.pyplot as plt

def load_image(image_path, size=None):
    target = cv2.imread(image_path, cv2.IMREAD_COLOR)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    if size:
        target = cv2.resize(src=target, dsize=size, interpolation=cv2.INTER_AREA)
    return target

def show_image(img_arr):
    plt.figure()
    plt.axis("off")
    plt.imshow(img_arr / 255)
    plt.show()