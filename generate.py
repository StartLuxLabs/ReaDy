"""Send a single thinking-mode request using the target tokenizer."""
import argparse
import json
import requests
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--target', required=True)
p.add_argument('--prompt', required=True)
p.add_argument('--url', default='http://127.0.0.1:30000')
p.add_argument('--max-new-tokens', type=int, default=1024)
a = p.parse_args()
tokenizer = AutoTokenizer.from_pretrained(a.target, local_files_only=True)
ids = tokenizer.apply_chat_template([{'role':'user', 'content':a.prompt}], tokenize=True,
                                   add_generation_prompt=True, enable_thinking=True, return_dict=False)
r = requests.post(a.url+'/generate', json={'input_ids':ids, 'sampling_params':{
    'temperature':0, 'top_p':1, 'top_k':-1, 'min_p':0, 'max_new_tokens':a.max_new_tokens}}, timeout=900)
r.raise_for_status()
print(json.dumps(r.json(), ensure_ascii=False, indent=2))
