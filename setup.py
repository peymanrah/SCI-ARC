"""
SCI-ARC: Structural Causal Invariance for Abstract Reasoning Corpus

A novel approach combining Structural Causal Invariance (SCI) principles
with Tiny Recursive Models (TRM) for the ARC benchmark.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sci-arc",
    version="0.1.0",
    author="SCI-ARC Team",
    author_email="sci-arc@example.com",
    description="Structural Causal Invariance for Abstract Reasoning Corpus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peymanrah/SCI-ARC",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sci-arc-train=scripts.train:main",
            "sci-arc-evaluate=scripts.evaluate:main",
        ],
    },
)
