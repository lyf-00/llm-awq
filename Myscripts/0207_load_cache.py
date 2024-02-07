import os
import itertools
import pathlib

def eval_all(model, outputdir, sdpath, dev, lm=True, qa=True, oi=True):
    lm_cmd = f'''python eval-nlp/eval_LM.py \
    --model-name {model} \
    --load {sdpath} \
    --output-dir {outputdir} '''

    qa_cmd = f'''python eval-nlp/eval_lm_eval.py \
    --model-name {model} \
    --load {sdpath} \
    --output-dir {outputdir}'''

    mmlu_cmd = f'''python eval-nlp/eval_lm_eval.py \
    --model-name {model} \
    --load {sdpath} \
    --tasks mmlu \
    --num-fewshot 5 \
    --output-dir {outputdir}'''
    if lm:
        print(f'CUDA_VISIBLE_DEVICES={dev} ' + lm_cmd)
    if qa:
        print(f'CUDA_VISIBLE_DEVICES={dev} ' + qa_cmd)
    if oi:
        print(f'CUDA_VISIBLE_DEVICES={dev} ' + mmlu_cmd)
    # if lm:
    #     os.system(f'CUDA_VISIBLE_DEVICES={dev} ' + lm_cmd)
    # if qa:
    #     os.sysmtem(f'CUDA_VISIBLE_DEVICES={dev} ' + qa_cmd)
    # if oi:
    #     os.system(f'CUDA_VISIBLE_DEVICES={dev} ' + oi_cmd)

# template = '''python -m awq.entry --model_path facebook/{} \
#     --w_bit {} \
#     --q_group_size 128 --run_awq \
#     --dump_awq awq_cache/{} \
#     --save_path {}'''
template = '''python -m awq.entry --model_path facebook/{} \
    --w_bit {} \
    --q_group_size 128  \
    --load_awq awq_cache/{} \
    --q_backend fake \
    --save_path {}'''
models = ['opt-125m','opt-1.3b','opt-2.7b','opt-6.7b','opt-13b','opt-30b']
wbits = [4,3,2]
dir_template = 'Experiments/{}-w{}-g128/'

dev = 0
for w_bit, model in itertools.product(wbits,models):
    dir = dir_template.format(model,w_bit)
    save_path = os.path.join(dir,'qsd.pt')
    # pathlib.Path(dir).mkdir(exist_ok=True,parents=True)
    print(f'mkdir -p {dir}')
    print(f'CUDA_VISIBLE_DEVICES={dev} ' + template.format(model,w_bit,f'{model}-w{w_bit}-g128.pt',save_path)+' >> ' + os.path.join(dir,'log.txt'))
    eval_all('facebook/'+model, dir, save_path, dev, oi=False)
    # os.system(template.format(model,w_bit,save_path)+' >> ' + os.path.join(dir,'log.txt'))
    dev = (dev + 1) % 8
    print('\n')


# template = '''python -m awq.entry --model_path {} \
#     --w_bit {} \
#     --q_group_size 128 --run_awq \
#     --dump_awq awq_cache_run/{} \
#     --save_path {}'''
template = '''python -m awq.entry --model_path {} \
    --w_bit {} \
    --q_group_size 128 \
    --load_awq awq_cache/{} \
    --save_path {}'''
models = ['meta-llama/Llama-2-7b-hf','meta-llama/Llama-2-13b-hf','meta-llama/Llama-2-70b-hf','yahma/llama-7b-hf','yahma/llama-13b-hf','~/yifei/llama-hf/30B','~/yifei/llama-hf/65B']
wbits = [4,3,2]
dir_template = 'Experiments/{}-w{}-g128'

model2name = {
    'meta-llama/Llama-2-7b-hf':'llama-2-7b',
    'meta-llama/Llama-2-13b-hf':'llama-2-13b',
    'meta-llama/Llama-2-70b-hf':'llama-2-70b',
    'yahma/llama-7b-hf':'llama-7b',
    'yahma/llama-13b-hf':'llama-13b',
    '~/yifei/llama-hf/30B':'llama-30b',
    '~/yifei/llama-hf/65B':'llama-65b'
}

dev = 0
for w_bit, model in itertools.product(wbits,models):
    dir = dir_template.format(model2name[model],w_bit)
    save_path = os.path.join(dir,'qsd.pt')
    # pathlib.Path(dir).mkdir(exist_ok=True,parents=True)
    print(f'mkdir -p {dir}')
    print(f'CUDA_VISIBLE_DEVICES={dev} ' + template.format(model,w_bit,f'{model2name[model]}-w{w_bit}-g128.pt',save_path)+' >> ' + os.path.join(dir,'log.txt'))
    eval_all(model, dir, save_path, dev)
    # os.system(template.format(model,w_bit,dir)+' >> ' + os.path.join(dir,'log.txt'))
    dev = (dev + 1) % 8
    print('\n')
    