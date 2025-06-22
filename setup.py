"""
Setup script for LLM Dataset Instruction Generator.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="llm-dataset-instruct-gen",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for generating instruction datasets from PDF documents using transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-dataset-instruct-gen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "accelerate>=0.20.0",
        ],
        "sentence-transformers": [
            "sentence-transformers>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-dataset-gen=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    keywords="llm dataset instruction generation pdf processing qa pairs transformers",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-dataset-instruct-gen/issues",
        "Source": "https://github.com/yourusername/llm-dataset-instruct-gen",
        "Documentation": "https://github.com/yourusername/llm-dataset-instruct-gen#readme",
    },
) 