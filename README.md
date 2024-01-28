# CGAT 
Attention-based Knowledge Graph Completion Combined with Contextualized Embedding

Requirements:
- python 3.8
- requirements.txt: `pip install -r requirements.txt`
- pytorch: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

Before start:
- run `prepare.sh`
- put the data at `data` directory

General usage:
- Set all configs at: `config.yml`
- Build geotrend and emvista dataset from the raw data: `python src/data_builder.py --dataset gt --split_ratio 0.1 0.1`
- Train Graph Embedding model: `python src/embeddings.py --dataset gt --model TransE --epochs 500 --dim 128`
- Train Contextualized Embedding model: `python src/contextualized.py --dataset gt --model bert-base-uncased --epochs 20 --dim 128`
- Train encoder: `python src/main.py -data gt -emb TransE -mode train_gat`
- Train decoder: `python src/main.py -data gt -emb TransE -mode train_conv`
- Evaluate: `python src/main.py -data gt -emb TransE -mode eval`