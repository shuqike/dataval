"""Test the time
data points 50
max epoch 5
err 0.1
"""
import sys
sys.path.append('../')
import time
from datasets import load_dataset
import transformers
transformers.logging.set_verbosity_warning()
from transformers import AutoTokenizer
from src.frameworks import TruncatedMC


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
dshap = TruncatedMC(
    train_dataset=small_train_dataset, 
    X_train=small_train_dataset['text'], 
    X_test=small_eval_dataset['text'], 
    test_dataset=small_eval_dataset, 
    model_family='bert-base-cased', 
    num_labels=5
)
print('start time', time.asctime())
dshap.run(save_every=100, err=0.1, do_loo=True, do_tmc=False, do_gshap=False)
print('end time', time.asctime())
