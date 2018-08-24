from setuptools import setup
from os import path

root_dir = path.abspath(path.dirname(__file__))


with open(path.join(root_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='peewee-migrations',
    version='0.3.10',
    url='https://github.com/aachurin/peewee_migrations',
    license='LGPL3',
    author='Andrey Churin',
    author_email='aachurin@gmail.com',
    description='Migration engine for peewee orm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['peewee_migrations'],
    zip_safe=False,
    platforms='any',
    install_requires=[
        'peewee >= 3.6.4',
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
    ],
    entry_points={
        'console_scripts': [
            'pem = peewee_migrations.cli:run'
        ],
    }
)
