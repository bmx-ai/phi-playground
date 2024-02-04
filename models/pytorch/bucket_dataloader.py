import random
from collections import defaultdict
import torch


class BucketDataLoader:
    
    @classmethod
    def from_dataset(cls, ds, tokenizer, get_text):
        buckets = defaultdict(list)

        for ex in ds:
            txt = get_text(ex) 
            toks = tokenizer([txt], padding=False, return_tensors="pt")
            token_len = len(toks[0])
            buckets[token_len].append(ex)
        return buckets  
    
    def __init__(self, dataset, batch_size=32, drop_last=False):
        
        
        
        buckets = defaultdict(list)
        for i, length in enumerate(lengths):
            if length > bmin:
                bucket_size = min((length // bstep) * bstep, bmax)
                buckets[bucket_size].append(i)
                
        self.buckets = dict()
        for bucket_size, bucket in buckets.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket, dtype=torch.int, device='cpu')
            
    def __iter__(self):
        
        if self.shuffle == True:
            for bucket_size in self.buckets.keys():
                self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]
                
        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket, self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket
            
        self.length = len(batches)
        
        if self.shuffle == True:
            random.shuffle(batches)
            
        return iter(batches)
    
    def __len__(self):
        return self.length