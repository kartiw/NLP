
In order to be able to import the score alignment file in our notebook, we changed the file name from score-alignment.py to score_alignment.py. We ran into an issue while trying to import the file due to the "fr" argument (mentioned in one of the issues on Discussion) and to solve that, we put the language and the output file name directly in the notebook. To test the code for German sentences, change the "language" argument to "de" and the "file" argument to "europarl_align.a" in the notebook. We also put the score code in the score_alignment.py file in a function, so that it could be called from the notebook. 


