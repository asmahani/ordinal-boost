from setuptools import setup, find_packages

setup(
    name='gbor',
    version='0.1.0',
    description='Gradient Boosting Ordinal Regression',
    author='Alireza S. Mahani',
    author_email='alireza.s.mahani@gmail.com',
    url='https://github.com/asmahani/ordinal-boost',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'scikit-learn>=0.22.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
