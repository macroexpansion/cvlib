from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    name="vtccvision",
    version="0.0.1",
    author="quangnd33",
    author_email="quangnd33@cyberspace.vn",
    description="A computer vision package",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "scikit-image",
    ],
)
