#!/usr/bin/env python3

"""Converting a series of png files in a folder to a gif animation.
"""

import argparse

import imageio


def images_2_gif(args):
    """ The main function to convert a series of pictures into a gif. """
    images = []
    with open(args.input_pics, "r") as f:
        for l in f:
            images.append(imageio.imread(l.strip("\n")))

    fps = args.fps
    duration = args.duration
    if fps is None:
        fps = 10
    if duration is None:
        duration = 1 / fps

    imageio.mimwrite(args.output_gif, images, "GIF", fps=fps, duration=duration)


def parse_arguments():
    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        type=str,
        dest="input_pics",
        help="A file consists of pics path with each pic on a single line.",
    )
    parser.add_argument("-o", type=str, dest="output_gif", help="Output gif path.")
    parser.add_argument("-fps", type=float, dest="fps", help="FPS.")
    parser.add_argument(
        "-duration", type=float, dest="duration", help="Duration of each frame."
    )

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_arguments()
    images_2_gif(ARGS)
