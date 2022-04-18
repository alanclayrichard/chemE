# Math Tools
This package is comprised of various math tools that I found myself using fairly often. The tools are separated into classes that categorize the methods into groups/topics in the math.py file. 

Quick Installation: (using pip3)

'''pip3 install git+https://github.com/alanclayrichard/math.git'''

Use in python:

import math_tools.math as math

predicted_model = math.Regression(x_train,y_train,1,x_test)

*This package is highly dependent on [numpy](https://numpy.org) and [pandas](https://pandas.pydata.org) and also uses [tqdm](https://github.com/tqdm/tqdm) in some for loops*

(set up from [this](https://towardsdatascience.com/create-your-custom-python-package-that-you-can-pip-install-from-your-git-repository-f90465867893) guide)
