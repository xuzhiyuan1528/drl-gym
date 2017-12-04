from skimage import color, transform
import matplotlib.pyplot as plt

def pre_process_image(image, width, height):
    # cut border
    image = image[30:195,:,:]
    # to grey scale
    image = color.rgb2grey(image)
    # resize
    image = transform.resize(image, (width, height), mode='reflect')
    return image

def pre_process_image_catcher(image):
    return color.rgb2grey(image)

def pre_process_image_bird(image, width, height):
    image = image[:400, :, :]
    image = color.rgb2grey(image)
    image = transform.resize(image, (width, height), mode='reflect')
    return image
    # plt.imshow(image, cmap='gray')
    # plt.show()