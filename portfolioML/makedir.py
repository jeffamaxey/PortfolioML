import argparse
import logging
import os
import shutil


def go_up(level_up):
    '''
    Simple function that returns the path of parent directory at a specified level

    Parameters
    ----------
    level up : int
        How much level you want to go up

    Returns
    --------
    None
    '''
    if level_up == 0:
        path = os.getcwd()
    if level_up == 1:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if level_up == 2:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    if level_up == 3:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    if level_up == 4:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    return path


def smart_makedir(name_dir, level_up=0):
    '''
    Easy way to create a folder. You can go up from the current path up to 4 times.

    Parameters
    ----------
    name_dir : str
        From level you have set, complete path of the directory you want to create
    level_up : TYPE, optional
        How many step up you want to do from the current path. The default is 0.

    Returns
    -------
    None.

    '''

    if level_up == 0:
        path = go_up(0) + "/" + name_dir
    if level_up == 1:
        path = go_up(1) + "/" + name_dir
    if level_up == 2:
        path = go_up(2) + "/" + name_dir
    if level_up == 3:
        path = go_up(3) + "/" + name_dir
    if level_up == 4:
        path = go_up(4) + "/" + name_dir

    if os.path.exists(path):
        logging.info(
            f"Path '{path}' already exists, it will be overwritten \n")
        # Remove all the files in case they already exist
        shutil.rmtree(path)
    os.makedirs(path)
    logging.info(f"Successfully created the directory '{path}' \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Easy way to create folders')
    parser.add_argument('name_directory', type=str,
                        help='Name of the directory that will be created')
    parser.add_argument('-level_up', default=0,
                        help='How many step up yo want from the current directory')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level=levels[args.log])

    smart_makedir(args.name_directory, args.level_up)
