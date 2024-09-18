---
title: Setup
---
# Requirements

## Software

You will need a terminal, Python 3.8+, and the ability to create Python virtual environments.

### Optional

It's recommended to have a code editor or Integrated Development Environment. Examples include **Spyder**, **VSCode/Visual Studio Code** or **PyCharm**. [Visual Studio Code](https://code.visualstudio.com/download) is lightweight and easy to install, and available on Window, Mac and Linux.


# Setup

Create a new directory for the workshop, then launch a terminal in it:

~~~
mkdir workshop-ml
cd workshop-ml
~~~
{: .language-bash}

## Creating a new Virtual Environment
You will need the **Numpy**, **Pandas**, **MatPlotLib**, **Seaborn** and **OpenCV** packages. 
We'll install these prerequisites in a virtual environment, to prevent them from cluttering up your Python environment or conflicting with any specific versions you have installed already.

To create a new virtual environment for the project, open the terminal and type:

~~~
python3 -m venv venv
~~~
{: .language-bash}

> ## Missing Module?
> If you're on Linux and this doesn't work, try installing `python3-venv` using your package manager, e.g. `sudo apt-get install python3-venv`.
{: .callout}

## Installing your prerequisites

Activate your virtual environment, and install the prerequisites:

~~~
source venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python
~~~
{: .language-bash}

You'll need to activate the environment again to use it at the start of the lesson.

{% include links.md %}
