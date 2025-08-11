from setuptools import setup, find_namespace_packages

setup(
    name = 'quantum-like-bits',
    version = '0.1.0',   
    description = "QL-bit simulation",
    long_description="""
        Research software code to study emergent states of QL-bits.
        """,
    url = 'https://github.com/Matematija/QubitRBM',
    author = 'William Horvat',
    author_email = 'whorvat@g.ucla.edu',
    license = 'MIT License',
    packages = find_namespace_packages(),
    install_requires = [
        'matplotlib==3.10.5',
        'networkx==3.2.1',
        'numpy==2.3.2'
    ],
    classifiers = [
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)