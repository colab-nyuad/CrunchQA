Download Crunchbase dump and unzip into folder data

Command to generate KG: python construct_kg.py

Command to shuffle and split generated triples into train, valid, test: to do (add bash script)

Command to train kg embeddings using pykeen: to do (add simple train file and add bash file with commands indicating parameters we used to compute our checkpoints)

Command to generate QA datset: to do (change in config.json names of columns and files according to the templates file, add the templates file, add one qa dataset generation file .py instead of all ipynb files)

Command to run KGQA framework: to do (add code for KGQA, with loading from pykeen checkpoint, changing dataloader for questions, minibatching (we cannot load the full matrix of embeddings))

Add file for qualitative evaluation: TBD
