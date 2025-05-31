# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="connection-transformer",
    version="1.0.0",
    author="Connection Transformer Research Team",
    author_email="research@connection-transformer.org",
    description="Bilinear Connections for Adaptive Reasoning in Transformer Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/connection-transformer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "wandb": [
            "wandb>=0.13.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.15.0",
            "ipywidgets>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "conn-trans-train=main:main",
            "conn-trans-analyze=analyze_results:main",
        ],
    },
    include_package_data=True,
    package_data={
        "configs": ["*.py"],
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="transformer, reasoning, bilinear, adaptive, neural networks, nlp",
    project_urls={
        "Bug Reports": "https://github.com/your-org/connection-transformer/issues",
        "Source": "https://github.com/your-org/connection-transformer",
        "Documentation": "https://connection-transformer.readthedocs.io/",
    },
)