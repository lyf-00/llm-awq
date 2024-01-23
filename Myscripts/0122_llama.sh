CUDA_VISIBLE_DEVICES=7 python -m awq.entry --model_path ./llama2-hf/llama-2-7b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-7b-w4-g128.pt     --save_path Experiments/llama-2-7b-w4-g128/qsd.pt >> Experiments/llama-2-7b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w4-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w4-g128 
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w4-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w4-g128
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w4-g128/qsd.pt     --tasks mmlu     --num-fewshot 5     --output-dir Experiments/llama-2-7b-w4-g128


CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ./llama2-hf/llama-2-13b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-13b-w4-g128.pt     --save_path Experiments/llama-2-13b-w4-g128/qsd.pt >> Experiments/llama-2-13b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w4-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w4-g128 
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w4-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w4-g128
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w4-g128/qsd.pt     --tasks mmlu     --num-fewshot 5     --output-dir Experiments/llama-2-13b-w4-g128


CUDA_VISIBLE_DEVICES=2 python -m awq.entry --model_path ./llama2-hf/llama-2-7b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-7b-w3-g128.pt     --save_path Experiments/llama-2-7b-w3-g128/qsd.pt >> Experiments/llama-2-7b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w3-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w3-g128 
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w3-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w3-g128
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w3-g128/qsd.pt     --tasks mmlu     --num-fewshot 5     --output-dir Experiments/llama-2-7b-w3-g128


CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path ./llama2-hf/llama-2-13b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-13b-w3-g128.pt     --save_path Experiments/llama-2-13b-w3-g128/qsd.pt >> Experiments/llama-2-13b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w3-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w3-g128 
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w3-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w3-g128
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w3-g128/qsd.pt     --tasks mmlu     --num-fewshot 5     --output-dir Experiments/llama-2-13b-w3-g128


CUDA_VISIBLE_DEVICES=4 python -m awq.entry --model_path ./llama2-hf/llama-2-7b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-7b-w2-g128.pt     --save_path Experiments/llama-2-7b-w2-g128/qsd.pt >> Experiments/llama-2-7b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w2-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w2-g128 
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w2-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w2-g128
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w2-g128/qsd.pt     --tasks mmlu     --num-fewshot 5     --output-dir Experiments/llama-2-7b-w2-g128


CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path ./llama2-hf/llama-2-13b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-13b-w2-g128.pt     --save_path Experiments/llama-2-13b-w2-g128/qsd.pt >> Experiments/llama-2-13b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w2-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w2-g128 
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w2-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w2-g128
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w2-g128/qsd.pt     --tasks mmlu     --num-fewshot 5     --output-dir Experiments/llama-2-13b-w2-g128


