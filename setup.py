import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='chemE_tools',
    version='0.0.7-beta.7',
    author='Clay Richard',
    author_email='alanclayrichard@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alanclayrichard/chemE',
    license='MIT',
    packages=['chemE_tools'],
    install_requires=['requests'],
)
