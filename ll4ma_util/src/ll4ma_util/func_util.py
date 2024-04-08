import os
import sys
import time
import argparse
import importlib  # For dynamic imports
from dataclasses import dataclass, fields, field
from typing import Dict, List
from contextlib import contextmanager

from ll4ma_util import file_util


@dataclass
class GenericDataClass:
    """
    Convenience dataclass to hold data from arbitrary dictionaries. Shortens attr access syntax.
    """
    def from_dict(self, dict_):
        for k, v in dict_.items():
            if isinstance(v, dict):
                setattr(self, k, dict_to_dataclass(v))
            else:
                setattr(self, k, v)

    def to_dict(self):
        return vars(self)


def dict_to_dataclass(dict_):
    dclass = GenericDataClass()
    dclass.from_dict(dict_)
    return dclass


def all_not_none(*args):
    return all([a is not None for a in args])


@contextmanager
def suppress_stdout():
    """
    Suppresses printing to stdout within the context. Useful when there
    is annoying terminal printing you can't seem to prevent.

    Usage:
        with suppress_stdout():
            print("This won't show up in terminal")

    Credit: http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def lambda_field(value):
    return field(default_factory=lambda: value)


@dataclass
class Config:
    """
    Base dataclass for configs.
    """
    # These fields are a workaround so you can directly instantiate configs by passing
    # a dictionary or YAML filename without having to first instantiate then call
    # from-function. If there's a better way to pass additional kwargs to dataclass
    # that would be preferable and can then get rid of __post_init__
    config_dict: Dict = field(default_factory=dict)
    config_filename: str = ''

    def __post_init__(self):
        if self.config_dict:
            self.from_dict(self.config_dict)
            self.config_dict = {}  # We don't need it anymore once dataclass is populated
        elif self.config_filename:
            self.from_yaml(self.config_filename)

    def to_dict(self):
        """
        Serialize config to dictionary.
        """
        dict_ = {}
        for field_ in fields(self):
            name = field_.name
            value = getattr(self, name)
            if isinstance(value, Config):
                dict_[name] = value.to_dict()
            elif isinstance(value, Dict):
                dict_[name] = {k: v.to_dict() if isinstance(v, Config) else v
                               for k, v in value.items()}
            elif isinstance(value, List):
                dict_[name] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                dict_[name] = value

            if 'config_dict' in dict_ and not dict_['config_dict']:
                del dict_['config_dict']
            if 'config_filename' in dict_ and not dict_['config_filename']:
                del dict_['config_filename']

        return dict_

    def from_dict(self, dict_):
        """
        Deserialize config from dictionary.
        """
        for field_ in fields(self):
            name = field_.name
            if name not in dict_:
                continue
            current_value = getattr(self, name)
            if isinstance(current_value, Config):
                current_value.from_dict(dict_[name])
            else:
                setattr(self, field_.name, dict_[field_.name])

    def to_yaml(self, filename):
        """
        Serialize config to YAML file.
        """
        dict_ = self.to_dict()
        file_util.save_yaml(dict_, filename)

    def from_yaml(self, filename):
        """
        Deserialize config from YAML file.
        """
        dict_ = file_util.load_yaml(filename)
        self.from_dict(dict_)


def override_dataclass_with_args(dc, args):
    """
    Anything set on args (parsed from argparse.ArgumentParser) will be set on
    the dataclass (dc) if its value is not None in args.
    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    elif not isinstance(args, dict):
        raise ValueError(f"Unknown input type for overriding dataclass values: {type(args)}")
    for k, v in args.items():
        if v is not None and hasattr(dc, k):
            setattr(dc, k, v)


def get_class(class_type, modules):
    """
    Returns the specified class imported from the provided modules. This is
    useful if you want to dynamically create an instance of a class knowing
    only the string name of the class and that it can be imported from one
    of the modules in the list. It will keep trying to import from each of
    the provided modules until it finds it.

    The returned class can be directly instantiated, e.g.:

        MyClass = get_class('MyClass', ['my_classes'])
        my_instance = MyClass()

    Args:
        class_type (str): String of the class to be retrieved
        modules (List[str]): List of modules from which imports of the class
                             should be attempted.
    Returns:
        The class requested, or None if it could not be imported from the
        provided modules.
    """
    Class = None
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        Class = getattr(module, class_type, None)
        if Class is not None:
            break
    return Class


def get_module_path(module):
    return os.path.dirname(module.__file__)


def silent_import(module_name):
    """
    Import a module by name without raising error if not found. Returns
    module if found, otherwise None.
    """
    module = None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass
    return module


class Timer:

    def __init__(self, start=False):
        if start:
            self.start()

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.duration = time.time() - self.start
        print(f"Time: {self.duration:.5f}")

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__()


if __name__ == '__main__':
    with Timer():
        time.sleep(2)
