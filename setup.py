import sys
from os import path

from notebuild.tool import get_version
from setuptools import find_packages, setup

version_path = path.join(path.abspath(
    path.dirname(__file__)), 'script/__version__.md')

version = get_version(sys.argv, version_path, step=16)

install_requires = ['notelbm']

setup(name='notelbm',
      version=version,
      description='notelbm',
      author='niult',
      author_email='1007530194@qq.com',
      url='https://github.com/1007530194',

      packages=find_packages(),
      install_requires=install_requires
      )
