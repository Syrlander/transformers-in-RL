from setuptools import setup, find_packages

setup(name='rl_thesis',
    version='0.1',
    description='RL thesis',
    author='Simon Surland Andersen, Emil Møller Hansen',
    author_email='glq414@alumni.ku.dk,ckb257@alumni.ku.dk',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
        'rl_thesis=rl_thesis.bin.rl_thesis:entry_func'
    ]
})