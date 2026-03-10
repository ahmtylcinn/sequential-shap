import os
from setuptools import setup, find_packages

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="sequential-shap-explainer",
    version="0.1.3",
    author="Ahmet Yalcin", 
    author_email="ahmtylcinn15@gmail.com", 
    description="An innovative approach to improve the interpretability of the SHAP method for sequential multi-class classification.", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ahmtylcinn/sequential-shap",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)