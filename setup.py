import os
import re
from setuptools import setup


def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


def get_long_description(long_description_file):
    with open(long_description_file, encoding='utf-8') as f:
        long_description = f.read()

    return long_description


version = get_version('peewee_migrations')


setup(
    name='peewee-migrations',
    version=version,
    url='https://github.com/aachurin/peewee_migrations',
    license='LGPL3',
    author='Andrey Churin',
    author_email='aachurin@gmail.com',
    description='Migration engine for peewee orm',
    long_description=get_long_description('README.md'),
    long_description_content_type='text/markdown',
    packages=['peewee_migrations'],
    zip_safe=False,
    platforms='any',
    install_requires=[
        'peewee >= 3.6.4',
        'click >= 7.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={
        'console_scripts': [
            'pem = peewee_migrations.cli:run'
        ],
    }
)
