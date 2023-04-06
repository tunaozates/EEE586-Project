#%%
from transformers import GraphormerForGraphClassification,GraphormerConfig,GraphormerModel
conf = GraphormerConfig(num_classes=2)
model = GraphormerForGraphClassification(conf)
#%%
from datasets import load_dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

graphs_dataset = load_dataset("graphs-datasets/ogbg-molhiv")
#%%
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool
from torch_geometric.transforms import ToUndirected

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='politifact',
                    choices=['politifact', 'gossipcop'])
parser.add_argument('--feature', type=str, default='spacy',
                    choices=['profile', 'spacy', 'bert', 'content'])
parser.add_argument('--model', type=str, default='GCN',
                    choices=['GCN', 'GAT', 'SAGE'])
args = parser.parse_args()

train_dataset = UPFD('politifact', args.dataset,'profile', 'train')
val_dataset = UPFD('politifact', args.dataset, 'profile', 'val')
test_dataset = UPFD('politifact', args.dataset, 'profile', 'test')
from datasets import Dataset
train_list = []
val_list = []
val_dict = {"edge_index":[],"y":[],"num_nodes":[],"node_feat":[]}
train_dict = {"edge_index":[],"y":[],"num_nodes":[],"node_feat":[]}
for graph in range(len(train_dataset)):
    train_dict["edge_index"].append(train_dataset[graph]["edge_index"])
    train_dict["y"].append(train_dataset[graph]["y"])
    train_dict["node_feat"].append(train_dataset[graph]["x"])
    train_dict["num_nodes"].append(train_dataset[graph]["x"].shape[0])
for graph in range(len(val_dataset)):
    val_dict["edge_index"].append(val_dataset[graph]["edge_index"])
    val_dict["y"].append(val_dataset[graph]["y"])
    val_dict["node_feat"].append(val_dataset[graph]["x"])
    val_dict["num_nodes"].append(val_dataset[graph]["x"].shape[0])
hf_train = Dataset.from_dict(train_dict)
hf_val =  Dataset.from_dict(val_dict)
#%%
maxx = 0
k=0
for g in range(len(hf_val)):
    if len(hf_val[g]["edge_index"][1])>maxx:
        maxx =len(hf_val[g]["edge_index"][1])
        k=g
print(maxx)
#%%
### Preprocessing
#import torch
#graphs_dataset = graphs_dataset.type(torch.LongTensor)
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
train_processed = hf_train.map(preprocess_item, batched=False)
val_processed = hf_val.map(preprocess_item, batched=False)
#%%

## Model

### Loading

from transformers import GraphormerForGraphClassification,GraphormerConfig 
conf=GraphormerConfig(num_classes=2,num_nodes=492,multi_hop_max_dist=5)
#conf=GraphormerConfig(num_classes=2,max_num_nodes=492,num_edges=1,num_attention_heads=2)
model = GraphormerForGraphClassification(
    conf
)
# model = GraphormerForGraphClassification.from_pretrained(
#     "clefourrier/pcqm4mv2_graphormer_base",
#     num_classes=2, # num_classes for the downstream task 
#     ignore_mismatched_sizes=True,
# )
#%%
maxx = 0
for i in range(len(hf_val)):
    k = hf_val[i]["num_nodes"] 
    if k > maxx:
        maxx = k
#%%
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    "graph-classification",
    #logging_dir="graph-classification",
    report_to="none",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    #auto_find_batch_size=True, # batch size can be changed automatically to prevent OOMs
    gradient_accumulation_steps=10,
    num_train_epochs=20,
    no_cuda=False,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_processed,
    eval_dataset=val_processed,
    data_collator=GraphormerDataCollator()
)
train_results = trainer.train()