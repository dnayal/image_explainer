from setuptools import setup, find_packages

setup(
    name='image_captioner',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'chromadb',
        'sentence-transformers',
    ],
)