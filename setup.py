from setuptools import find_packages, setup, Extension, Command
from sys import executable

class BenchmarkCommand(Command):
    user_options = []
    description = "Benchmark this package"

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from subprocess import call, PIPE
        call([executable, 'setup.py', 'build_ext', '--inplace'], stdout=PIPE)
        call([executable, '-m', 'fastecdsa.benchmark'])


curvemath = Extension(
    'fastecdsa.curvemath',
    include_dirs=['src/'],
    libraries=['gmp'],
    sources=['src/curveMath.c', 'src/curve.c', 'src/point.c'],
    extra_compile_args=['-O2']
)

_ecdsa = Extension(
    'fastecdsa._ecdsa',
    include_dirs=['src/'],
    libraries=['gmp'],
    sources=['src/_ecdsa.c', 'src/curveMath.c', 'src/curve.c', 'src/point.c'],
    extra_compile_args=['-O2']
)

setup(
    author='Anton Kueltz',
    author_email='kueltz.anton@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Security :: Cryptography',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
    ],
    cmdclass={'benchmark': BenchmarkCommand},
    description='Fast elliptic curve digital signatures',
    ext_modules=[curvemath, _ecdsa],
    keywords='elliptic curve cryptography ecdsa ecc',
    license='CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
    long_description=''.join(open('README.rst', 'r').readlines()),
    name='fastecdsa',
    packages=find_packages(),
    url='https://github.com/AntonKueltz/fastecdsa',
    version='2.2.3',
)
