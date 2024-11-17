from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='InfoFuzzyNetwork',
    version='1.0.1',  # Package version
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        'numpy',  # Dependencies
        'pandas',
        'scipy',
        'networkx',
        'matplotlib',
        'scikit-learn'
    ],
    description='Implementation of the IFN (Info Fuzzy Network) model for predictions and feature selection on structured data.',
    long_description=(Path(__file__).parent / "README.md").read_text(encoding='utf-8'),
    long_description_content_type="text/markdown",
    author='Shahar Oded',
    author_email='shahar6771@gmail.com',
    url='https://github.com/shaharoded/IFN-Info-Fuzzy-Network',  # Optional: Link to your GitHub repo
    classifiers=[
    'Development Status :: 4 - Beta',  # Update to '5 - Production/Stable' if fully ready
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    license='MIT'
)