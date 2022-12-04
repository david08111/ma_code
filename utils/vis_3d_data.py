from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
import pickle


def show_3d_data(data_paths):

    list_colors = ["r", "g", "b", "c", "m", "y", "k", "w"]

    fig = plt.figure()  # Create 3D container
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(data_paths)):
        with open(data_paths[i], "rb") as file_handler:
            data = pickle.load(file_handler)
        file_handler.close()

        ax.scatter(data[:,0], data[:,1], data[:,2], c=list_colors[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--data_file", action="append",
                        help="Filepath to data")

    args = parser.parse_args()
    show_3d_data(args.data_file)