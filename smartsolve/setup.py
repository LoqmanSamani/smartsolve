from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.txt"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.4'
DESCRIPTION = 'A machine learning package'
LONG_DESCRIPTION = 'A machine learning package for data analysis and predictive modeling'

# Setting up
setup(
    name="smartsolve",
    version=VERSION,
    author="Loghman Samani",
    author_email="samaniloqman91@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=["pandas", "numpy", "scikit-learn", "scipy", "seaborn", "matplotlib"],
    keywords=['python', 'machine learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
