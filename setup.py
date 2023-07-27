from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    """The function `get_requirements` reads a file and returns a list of requirements, removing any
    occurrences of the "-e.".
    
    Param: file_path (str) -- The file path is a string that represents the path to the file containing the
    requirements
    
    returns: A list of requirements.
    """    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

#setup - project informations, name, version etc
# The `setup()` function is a function provided by the `setuptools` library in Python. It is used to
# define the metadata and configuration options for a Python package.

setup(
    name= "Dimond Price Prediction",
    version= "0.0.1",
    author= "Ashutosh Vaidya",
    author_email= "ashutosh.vaidya1190@gmail.com",
    install_requires= get_requirements("requirements.txt"),
    packages= find_packages()
)