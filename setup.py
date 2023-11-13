import codecs

from setuptools import find_packages, setup

DISTNAME = "Conformalized GP"
DESCRIPTION = (
    "Package for conformalized Gaussian process regression "
    "based on the MAPIE librairy."
)
with codecs.open("README.md", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/x-rst"
URL = "https://github.com/scikit-learn-contrib/MAPIE"
DOWNLOAD_URL = "https://pypi.org/project/MAPIE/#files"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/scikit-learn-contrib/MAPIE/issues",
    "Documentation": "https://mapie.readthedocs.io/en/latest/",
    "Source Code": "https://github.com/scikit-learn-contrib/MAPIE"
}
LICENSE = "new BSD"
MAINTAINER = "V. Blot, E. Jaber"
MAINTAINER_EMAIL = (
    "vblot@quantmetry.com, "
    "edgar.jaber@edf.fr"
)
PYTHON_REQUIRES = ">=3.8"
PACKAGES = find_packages()
INSTALL_REQUIRES = ["scikit-learn", "scipy", "numpy>=1.21", "packaging"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10"
]

setup(
    name=DISTNAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    download_url=DOWNLOAD_URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    classifiers=CLASSIFIERS,
    zip_safe=False  # the package can run out of an .egg file
)
