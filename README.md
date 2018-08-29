Science has made staggering advances in the last three decades. Perhaps inevitably, the language that researchers use to talk about science has also changed dramatically during this time. I want to see how changes in the scientific lexicon are reflected in the content of successful grant applications. What are the major trends that have come and gone over the years? Has the style of grant writing itself developed? Is it possible to predict which decade a grant was written just by looking at the abstract? 


## Data aquisition

I obtained a sample of grant abstracts from the NIH website using the script
[download_files.sh](RNN/download_files.sh)

## Analysis

Visualization of word usage and comparison between years using R can be found in the [analysis](analysis) directory.

## Deep learning experiment (RNN)

I built a [recurrent neural net (RNN)](RNN) that can predict which decade a grant abstract came from based on the text alone. I am currently working on a RNN that can generate grant abstracts.

### keras_classify.py
RNN for classifying abstracts by year

### run.sh
Run train.py on SLURM cluster

### generator.py
Text generator based on grant data
