# TAPI Topic Models

Notebooks for the NEH TAPI Workshop, "How to Do Things with Topic Models."

# Code

The code for these workshops are found in a collection of Jupyter Notebooks hosted on the Constellete Binder hub. To access them, click on the button below:

[![Binder](https://binder.constellate.org/badge_logo.svg)](https://binder.constellate.org/v2/gh/ontoligent/TAPI_Topic_Models/main)

# Corpus Data

Corpus data files are `CSV` files in a specific format &mdash; __machine learning corpus format__. Files in this format are essentially files with three columns: (1) a document identity, called `doc_key`, (2) a document label, called `doc_label`, and (3) the document content, called `doc_content`. Each row contains a complete "document," defined as an analytically useful unit of discourse, such as a paragraph or chapter. 

A collection of them have been prepared for this workshop. Download them and then upload them to the `./corpora` folder in your Binder repository. Sorry that the process is not more direct!

Corpus data may be downloaded from the following shared Dropbox link:

* [TAPI Corpora Directory](https://www.dropbox.com/sh/t6im8ni921gxinr/AADL_-VPetjmDIMO3vYAFvNRa?dl=0)  

Additionally, these files may be downloaded individually:

* [Wine Reviews](https://www.dropbox.com/s/0rszsd6t30c0n3y/winereviews-tapi.csv?dl=0) &mdash; A collection of terse wine reviews.
* [JSTOR Hyperparameter](https://www.dropbox.com/s/uoa8191px405fj0/jstor_hyperparameter-tapi.csv?dl=0) &mdash; Abstracts from a JSTOR search for "hyperparameter." 
* [Tamilnet](https://www.dropbox.com/s/dtqnzcbkcp07u5e/tamilnet-tapi.csv?dl=0) &mdash; A sample of news stories from the website Tamilnet.
* [Anphoblach](https://www.dropbox.com/s/lrmt92q59npx0x5/anphoblacht-tapi.csv?dl=0) &mdash; A sample of news stories from the website Anphoblacht.

Each link goes to a Dropbox item that has a download link. Download the file to your desktop and then upload to the appropriate directory.

# Sample Output Data

The notebooks in this workshop will generate a __digital analytical edition__ from a given source corpus file. The results of the various analytical processes will be put in the `./db` directory. A demonstration edition is provided for one of the notebooks. To get the demo data, download then upload the files with the prefix `jstor_hyperparameter_demo` into to your `./db` directory.

* [TAPI Editions Directory](https://www.dropbox.com/sh/51viylnaqlrcsy9/AAAfW8KcVu-PlU_APGXhDd-Va?dl=0)

# Workshop Slides

* [Day 1](https://docs.google.com/presentation/d/1v98_J4UErZqt6U4ut8DACBQa7aOtaBQFyl134cVGAG4/edit?usp=sharing)
* Day 2
* Day 3

<!--
.. image:: https://binder.constellate.org/badge_logo.svg
 :target: https://binder.constellate.org/v2/gh/ontoligent/TAPI_Topic_Models/main
-->