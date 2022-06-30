Download Crunchbase dump and unzip into folder data

Command to generate KG: python construct_kg.py

Command to shuffle and split generated triples into train, valid, test: ./kg/split_train_valid_test.bh

Command to train kg embeddings using pykeen:
python train_embeddings.py --model DistMult --train_path kg/clustering/train.txt --valid_path kg/vanilla/valid.txt --test_path kg/vanilla/test.txt --dim 200 --results_folder embeddings/clustering_distmult --gpu 1

Command to generate QA datset: python genrate_dataset.py

Command to shuffle and split generated questions into train, valid, test: ./qa_dataset/split_train_valid_test.bh

Command to run KGQA framework: 
python run.py --model DistMult --embeddings_folder embeddings/clustering_distmult --freeze True


Mapped Jobs:

"director": ["director"]
"CSO": ["chief security officer", "cso"] 
"board_member": ["chairman", "boar director", "board member"] 
"founder": ["founding partner", "founder", "co-founder", "co founder", "cofounder"] 
"CFO": ["chief financial officer", "cfo"] 
"CEO": ["chief executive officer", "ceo"] 
"CPO": ["chief people officer", "cpo"]
"CIO": ["chief information officer", "cio"] 
"CBO": ["chief business officer", "cbo"]
"VP": ["vice president", "vp"] 
"advisor": ["advisor"]
"owner": ["owner"]
"president": ["president"]
"CTO": ["chief technology offcier", "cto"]
"CRO": ["chief revenue officer", "cro"]
"COO": ["chief operating officer", "coo"]

