import numpy as np
import matplotlib.pyplot as mp
import matplotlib.animation as ma


def main():
    data_file = open("data/mnist_1_test_10_.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()

    fig = mp.figure()

    images = []
    for i in range(len(data_list)):
        values = data_list[i].split(',')
        image_array = np.asfarray(values[1:]).reshape((28, 28))
        image = mp.imshow(image_array, cmap='Greys', animated=True)
        images.append([image])

    anim = ma.ArtistAnimation(fig, images, interval=500, blit=False,
            repeat_delay=1000)

    mp.show()


main()
