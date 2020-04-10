import numpy as np
import matplotlib.pyplot as mp

def main():
    data_file = open("data/mnist_1_test_10_.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()

    images = []
    for i in range(len(data_list)):
        values = data_list[i].split(',')
        image_array = np.asfarray(values[1:]).reshape((28, 28))
        fig, ax = mp.subplots()
        images.append(ax.imshow(image_array, cmap='Greys'))

    mp.show()

main()
