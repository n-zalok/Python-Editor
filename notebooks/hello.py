"""
Docstring for Python-Editor.notebooks.hello
"""
import sys
from torch import nn

def get_name():
    """
    Docstring for get_name
    """
    # get name
    # asda
    got_name = input("Enter your name: ")
    return got_name

def greet(name):
    """
    Docstring for greet
    
    :param name: Description
    """
    print(f"Hello, {name}")

def main():
    name = get_name()
    greet(name)

if __name__ == "__main__":
    main()