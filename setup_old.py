from setuptools import setup, find_namespace_packages

setup(
    name='quantum-like-bits',
    version='0.1.0',
    description="QL-bit simulation",
    long_description="""
    Research software code to study emergent states of QL-bits.
    """,
    url='https://github.com/wahorvat/quantum-like-bits',
    author='William Horvat',
    author_email='whorvat@g.ucla.edu',
    license='MIT License',
    packages=find_namespace_packages(),
    install_requires=[
        'matplotlib>=3.7.0,<4.0.0',
        'networkx>=3.0.0,<4.0.0', 
        'numpy>=1.22.0,<2.0.0', 
        'scipy>=1.9.0,<2.0.0',
    ],
    python_requires='>=3.8,<4.0',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)