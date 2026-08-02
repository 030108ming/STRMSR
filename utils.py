"""
Small filesystem helpers shared by train.py and test.py.
"""

import os


def prepare_sub_folder(output_directory):
    """
    Create the images/ and checkpoints/ subfolders of an experiment directory.

    Returns:
        (checkpoint_directory, image_directory)
    """
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print(f"Creating directory: {image_directory}")
        os.makedirs(image_directory)

    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print(f"Creating directory: {checkpoint_directory}")
        os.makedirs(checkpoint_directory)

    return checkpoint_directory, image_directory
