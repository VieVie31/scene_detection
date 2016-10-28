#! /usr/bin/env python3

import argparse
from os import system
from sys import argv
from re import match

def resolution_parser(v):
    if not match(r'\dx\d', v):
        raise argparse.ArgumentTypeError('Incorrect resolution : format must be widthxheight')
    w, h = map(v.split('x'), int)
    print(w, h)
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError('Incorrect resolution : width and height must be positive integers')

    return v

def fps_parser(v):
    try:
        if int(v) < 1:
            raise argparse.ArgumentTypeError('Incorrect fps : must be positive')
        return v
    except:
        raise argparse.ArgumentTypeError('Incorrect fps : must be an integer')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="scd", description='Shrink videos to delete openings.')
    parser.add_argument('-f', '--fps', type=int, help='Number of frame per second analysed.', default=5)
    parser.add_argument('-r', '--resolution', type=str, help='Set analysis resolution: widthxheight.', default='8x8')
    parser.add_argument('-sl', '--sequence-length', type=int, help='[DEPRECATED] This param will be ignored. Get the hashes of video by sequences of `sl` frames.', default=1)  # TODO : implement
    parser.add_argument('-sw', '--sliding-window', type=int, help='Set inter sequences hashes step.', default=1)                       # TODO : implement
    parser.add_argument('-nc', '--no-cache', action='store_true', help='Disable application video encoding cache.')
    parser.add_argument('-cc', '--clear-cache', action='store_true', help='Clear all files cached before running the rest of the app.')

    args = parser.parse_args(argv[1:])

    print('Building container')
    # Build image ( Or use cache if avaliable )
    system('docker build -t scd .')

    print('Running container')

    # Ensuring volumes organization
    system('mkdir -p ./videos ./cache/stats')

    # run with videos in the volume...
    system('docker run  -v $(pwd)/src:/src \
                        -v $(pwd)/videos:/videos \
                        -v $(pwd)/cache:/cache \
                        -e "CACHE=%s" \
                        -e "CLEAR_CACHE=%s"\
                        -e "FPS=%s" \
                        -e "RESOLUTION=%s" \
                        -e "SEQUENCE_LENGTH=%s" \
                        -e "SLIDING_WINDOW=%s" \
                        -it \
                        scd' \
                        % (('0' if args.no_cache else '1'),
                           ('1' if args.clear_cache else '0'),
                            args.fps,
                            args.resolution,
                            args.sequence_length,
                            args.sliding_window))
