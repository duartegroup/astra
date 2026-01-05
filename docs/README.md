# Compiling ASTRA's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To compile the docs, first ensure that the necessary dependencies are installed.

You can install them using pip:

```bash
pip install -r requirements.txt
```

Once installed, you can use the `Makefile` in this directory to compile static HTML pages by running:
```bash
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html` (which may itself be inside a directory called `html/` depending on what version of Sphinx is installed).
