"""Setup script for MNIST Classifier."""

from setuptools import setup, find_packages

with open("../requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mnist-classifier",
    version="1.0.0",
    author="Claude Code Automation Framework",
    description="MNIST digit classification system with multiple model types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/mnist-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mnist-classifier=mnist_classifier.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "mnist_classifier": ["*.yml", "*.yaml", "*.json"],
    },
)