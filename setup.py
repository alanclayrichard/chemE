import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='math_tools',
    version='0.0.5',
    author='Clay Richard',
    author_email='alanclayrichard@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alanclayrichard/math',
#     project_urls = {
#         "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
#     },
    license='MIT',
    packages=['math_tools'],
    install_requires=['requests'],
)
