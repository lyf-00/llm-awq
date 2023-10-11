import os
import json
import pandas as pd
import numpy as np
import itertools
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tf_results(output_dir):
    if not os.path.exists(output_dir):
        print(output_dir)
        return None
    record = {}
    for file_name in os.listdir(output_dir):
        if 'events.out.tfevents.' in file_name:
            event_file = os.path.join(output_dir, file_name)
            accumulator = EventAccumulator(event_file)
            accumulator.Reload()
            for tag in ['eval/ppl_wikitext2', 'eval/ppl_ptb-new', 'eval/ppl_c4-new']:
                if tag in accumulator.Tags()['scalars']:
                    record[tag.split('_')[-1]] = accumulator.Scalars(tag)[-1].value  # Take the last value if multiple
    print(record)
    return record

def extract_txt_results(output_dir):
    try:
        with open(os.path.join(output_dir,'log.txt'),'r') as f:
            lines = f.read().split('\n')
        record = {}
        record['wikitext2'] = float(lines[lines.index('ptb-new')-1])
        record['ptb-new'] = float(lines[lines.index('c4-new')-1])
        try:
            record['c4-new'] = float(lines[-1])
        except:
            record['c4-new'] = float(lines[-2])
        print(record)
        return record
    except:
        return {}


models = ['opt-125m','opt-350m','opt-1.3b','opt-2.7b','opt-6.7b','opt-13b','opt-30b','./llama2-hf/llama-2-7b','./llama2-hf/llama-2-13b']
wbits = [4,3,2]
dir_template = 'Experiments/{}-w{}-g128/'

df = pd.DataFrame(columns=['model','wbit','wikitext2','ptb-new','c4-new','piqa','hellaswag','winogrande','arc_easy','arc_challenge'])
for model, wbit in itertools.product(models,wbits):
    if 'llama' in model:
        model = model.split('/')[-1]
    dir_path = dir_template.format(model,wbit)
    print(dir_path)
    # Language modeling tasks 
    res = {'model':model,'wbit':wbit}
    try:
        with open(os.path.join(dir_path,'LM_results.json'),'r') as f:
            res.update(json.load(f)) 
    except:
        # res.update(extract_tf_results(dir_path))
        res.update(extract_txt_results(dir_path))
    try:
        with open(os.path.join(dir_path,'QA_results_lm_eval.json'),'r') as f:
            qa = json.load(f)['results']
        for i in ['piqa','hellaswag','winogrande','arc_easy','arc_challenge']:
            res[i] = qa[i]['acc']
        res += qa
    except:
        pass
    # print(res)
    df_new = pd.DataFrame(res,index=[0])
    df = pd.concat([df,df_new])
    # print(df)
    # break
df.to_csv('results.csv')
