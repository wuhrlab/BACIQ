from setuptools import setup, find_packages

setup(
    name='BACIQ',
    version=0.1,
    description='Confidence to proteomics measurements',
    url='https://github.com/wuhrlab/BACIQ',
    packages=find_packages(),
    install_requires=[
        'Click',
        'pystan',
        'pandas'
    ],
    entry_points='''
        [console_scripts]
        baciq=baciq.baciq:main
''',
)
