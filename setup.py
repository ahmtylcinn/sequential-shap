from setuptools import setup, find_packages

setup(
    name="sequential-shap",
    version="0.1.0",
    author="Ahmet Yalcin", 
    author_email="ahmtylcinn15@gmail.com", 
    description="An innovative approach to improve the interpretability of the SHAP method for sequential multi-class classification.", 
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "shap"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)