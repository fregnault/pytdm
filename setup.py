import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()
with open('requirements.txt', 'r') as f:
    requirements = f.read().strip('\n').split('\n')

# package_data = {
    # '': ['samples/*', 'res_findcube/*'],
    # }

setuptools.setup(
    name='pytdm',
    version='1.0',
    author='Florian Regnault',
    author_email='fl.regnault@gmail.com',
    description='Python tool to initialize a Titov-Démoulin flux rope in a simulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fregnault/pyTDm.git',
    packages=setuptools.find_packages(),
    # package_data=package_data,
    python_requires='>=3.7',
    setup_requires=['wheel'],
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        # 'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'
        ]
    )
