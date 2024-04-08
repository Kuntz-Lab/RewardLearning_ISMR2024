import os
import os.path as osp
import yaml
import json
import pickle
import shutil
import xml.etree.ElementTree as ET
from glob import glob

from ll4ma_util import ui_util

PICKLE_EXTENSIONS = ['.pickle', '.pkl']

# Utility functions for working with files/directories and loading/saving
# data from various file formats (yaml, pickle).


def list_dir(directory, extension='', exclude_files=False, exclude_dirs=False):
    """
    Returns a list of contents in directory as absolute paths.

    Args:
        directory (str): Absolute path to list files for.
        extension (str): Optional file extension to filter with
        exclude_files (bool): Exclude files if True
        exclude_dirs (bool): Exclude directories if True
    Returns:
        contents (lst): List of absolute paths to contents (files and 
                        subdirectories) of directory.
    """
    directory = osp.expanduser(directory)
    contents = glob(osp.join(directory, '*')) if osp.exists else []
    if extension:
        contents = [c for c in contents if c.endswith(extension)]
    if exclude_files:
        contents = [c for c in contents if not osp.isfile(c)]
    if exclude_dirs:
        contents = [c for c in contents if not osp.isdir(c)]
    return contents


def clear_dir(directory):
    """
    Removes all files and sub-directories in directory.

    Args:
        directory (str): Absolute path to directory to clear.
    """
    for path in list_dir(directory):
        remove(path)


def remove_dir(directory):
    shutil.rmtree(directory)


def dir_of_file(file_, parent_level=0):
    """
    Returns the directory that contains a particular file (you should 
    pass in __file__ from the file you are wanting to get the containing 
    directory of). Specify parent_level to get higher up parents.

    For example, if I have /path/to/my/script.py, if I call

        dir_of_file(__file__)

    from script.py, it will return /path/to/my, and parent_level=1 will
    give /path/to, and parent_level=2 will give /path.
    """
    dir_ = osp.dirname(osp.abspath(file_))
    for _ in range(parent_level):
        dir_ = osp.dirname(dir_)
    return dir_


def remove(path):
    if osp.isfile(path):
        os.remove(path)
    elif osp.isdir(path):
        remove_dir(path)
        

def safe_remove_path(path):
    if osp.exists(path):
        ui_util.print_warning(f"\n  Path already exists: {path}")
        overwrite = ui_util.query_yes_no("  Overwrite this data?")
        if overwrite:
            ui_util.print_info(f"  Overwriting existing data.\n\n")
        else:
            ui_util.print_info_exit(f"\n  Exiting. Existing data is intact.\n\n")
    remove(path)

            
def create_dir(directory, exist_ok=True):
    os.makedirs(directory, exist_ok=exist_ok)

            
def safe_create_dir(directory, exit_on_skip=True):
    """
    Creates new directory, checks with user if they want to overwrite data
    if the directory already exists.

    Args:
        directory (str): Absolute path to directory to be created.
        exit_on_skip (str): Calls sys.exit if True and user does not want to overwrite,
                            otherwise prints message and continues.
    Return:
        Returns True if data was deleted and directory was created, False if existing
        data is left intact and we didn't exit on skip.
    """
    if not osp.isdir(directory):
        os.makedirs(directory)
    elif len(os.listdir(directory)) > 0:
        ui_util.print_warning(f"\n  Directory already exists: {directory}")
        overwrite = ui_util.query_yes_no("  Overwrite data in this directory?")
        if overwrite:
            clear_dir(directory)
            ui_util.print_info(f"  Existing data deleted. Writing new files to {directory}\n\n")
        elif exit_on_skip:
            ui_util.print_info_exit(f"\n  Exiting. Existing data is intact in {directory}.\n\n")
        else:
            ui_util.print_info(f"\n  Existing data is intact in {directory}.\n\n")
            return False
    return True

def force_create_dir(directory):
    if osp.exists(directory):
        remove_dir(directory)
    create_dir(directory)
            

def check_path_exists(path, msg_prefix="Path"):
    """
    Check if a path (file or directory) exists. If it does not, do a sys exit.

    This is a utility function that is useful for checking that files exist
    when parsing filenames from command line args.

    Args:
        path (str): Absolute path to check if it exists.
        msg_prefix (str): Descriptive prefix about that path to print a more
                          helpful error message if it exits (e.g. set 
                          msg_prefix='Config file' to better inform user which
                          file does not exist)
    """
    if not osp.exists(path):
        ui_util.print_error_exit(f"\n{msg_prefix} does not exist: {path}\n")


def copy_file(src, dest):
    """
    Copies file to a new location.

    Args:
        src (str): Absolute path of file to be copied.
        dest (str): Absolute path where file will be copied to.
    """
    shutil.copyfile(src, dest)


def move(src, dest):
    """
    Moves file/directory to a new location.

    Args:
        src (str): Absolute path of file/directory to be moved.
        dest (str): Absolute path where file/directory will be moved to.
    """
    shutil.move(src, dest)


def save_yaml(data, filename):
    """
    Saves data stored in a dictionary to a yaml file.

    Args:
        data (dict): Dictionary of data to be saved to yaml file.
        filename (str): Absolute path to yaml file that data will be saved to.
    """
    filename = osp.expanduser(filename)
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filename):
    """
    Loads data form a yaml file to a dictionary.

    Args:
        filename (str): Absolute path to yaml file to load data from.
    Returns:
        data (dict): Dictionary of data loaded from the yaml file.
    """
    filename = osp.expanduser(filename)
    check_path_exists(filename, "YAML file")
    with open(filename, 'r') as f:
        try:
            data = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            data = yaml.load(f)
    return data


def parse_dict_string(dict_str):
    """
    Uses yaml to parse string representation of dictionary to a dict.
    """
    dict_ = yaml.load(dict_str, Loader=yaml.FullLoader)
    return dict_


def save_pickle(data, filename):
    """
    Saves data stored in a dictionary to a pickle file.

    Args:
        data (dict): Dictionary of data to be saved to pickle file.
        filename (str): Absolute path to pickle file that data will be saved to.
    """
    filename = osp.expanduser(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename, check_extension=True):
    """
    Loads data from a pickle file to a dictionary.

    Args:
        filename (str): Absolute path to pickle file to load data from.
    Returns:
        data (dict): Dictionary of data loaded from the pickle file.
    """
    filename = osp.expanduser(filename)
    if check_extension and osp.splitext(filename)[1] not in PICKLE_EXTENSIONS:
        ui_util.print_error_exit(f"\nTried to load a pickle file with invalid extension: {filename}\n")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def load_json(filename):
    filename = osp.expanduser(filename)
    check_path_exists(filename, "JSON file")
    with open(filename) as f:
        data = json.load(f)
    return data


def load_xml(filename):
    """
    Load data from an XML file using ElementTree. See docs for details:
        
        https://docs.python.org/3/library/xml.etree.elementtree.html
    
    Args:
        filename (str): Absolute path to XML file to load data from.
    Returns:
        data (Element): An Element instance representing the data.
    """
    filename = osp.expanduser(filename)
    check_path_exists(filename, "XML file")
    root = ET.parse(filename).getroot()
    return root


def change_extension(filename, extension):
    """
    Changes the extension of the input filename to the desired extension.

    Args:
        filename (str): Filename to change extension for (assumes format NAME.EXTENSION)
                        where NAME can be any absolute or relative path.
    Returns:
        new_filename (str): Filename with the extension changed as specified
    """
    base = osp.splitext(filename)[0]
    new_filename = f"{base}.{extension}"
    return new_filename


def get_path(module, *args):
    """
    Returns a string of the absolute path of the module, with additional 
    sub-directories appended to the path specified through args.
    """
    return osp.join(osp.dirname(module.__file__), *args)
