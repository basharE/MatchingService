import os


def create_path(directory):
    # checking if the directory demo_folder
    # exist or not.
    if not os.path.exists(directory):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(directory)
