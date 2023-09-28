CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path facebook/opt-125m     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-125m-w4-g128.pt     --save_path Experiments/opt-125m-w4-g128/qsd.pt >> Experiments/opt-125m-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_LM.py     --model-name facebook/opt-125m     --load Experiments/opt-125m-w4-g128/qsd.pt     --output-dir Experiments/opt-125m-w4-g128/ 
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-125m     --load Experiments/opt-125m-w4-g128/qsd.pt     --output-dir Experiments/opt-125m-w4-g128/

CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path facebook/opt-350m     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-350m-w3-g128.pt     --save_path Experiments/opt-350m-w3-g128/qsd.pt >> Experiments/opt-350m-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_LM.py     --model-name facebook/opt-350m     --load Experiments/opt-350m-w3-g128/qsd.pt     --output-dir Experiments/opt-350m-w3-g128/ 
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-350m     --load Experiments/opt-350m-w3-g128/qsd.pt     --output-dir Experiments/opt-350m-w3-g128/

CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path facebook/opt-1.3b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-1.3b-w2-g128.pt     --save_path Experiments/opt-1.3b-w2-g128/qsd.pt >> Experiments/opt-1.3b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_LM.py     --model-name facebook/opt-1.3b     --load Experiments/opt-1.3b-w2-g128/qsd.pt     --output-dir Experiments/opt-1.3b-w2-g128/ 
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-1.3b     --load Experiments/opt-1.3b-w2-g128/qsd.pt     --output-dir Experiments/opt-1.3b-w2-g128/

CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path facebook/./llama2-hf/llama-2-7b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-7b-w4-g128.pt     --save_path Experiments/llama-2-7b-w4-g128/qsd.pt >> Experiments/llama-2-7b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w4-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w4-g128 
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w4-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w4-g128
CUDA_VISIBLE_DEVICES=0 python eval-nlp/eval_MMLU_oi.py     --model_name_or_path ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w4-g128/qsd.pt     --save_dir Experiments/llama-2-7b-w4-g128/MMLU_OI     --eval_batch_size 1

CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path facebook/opt-350m     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-350m-w4-g128.pt     --save_path Experiments/opt-350m-w4-g128/qsd.pt >> Experiments/opt-350m-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_LM.py     --model-name facebook/opt-350m     --load Experiments/opt-350m-w4-g128/qsd.pt     --output-dir Experiments/opt-350m-w4-g128/ 
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-350m     --load Experiments/opt-350m-w4-g128/qsd.pt     --output-dir Experiments/opt-350m-w4-g128/

CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path facebook/opt-1.3b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-1.3b-w3-g128.pt     --save_path Experiments/opt-1.3b-w3-g128/qsd.pt >> Experiments/opt-1.3b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_LM.py     --model-name facebook/opt-1.3b     --load Experiments/opt-1.3b-w3-g128/qsd.pt     --output-dir Experiments/opt-1.3b-w3-g128/ 
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-1.3b     --load Experiments/opt-1.3b-w3-g128/qsd.pt     --output-dir Experiments/opt-1.3b-w3-g128/

CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path facebook/opt-2.7b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-2.7b-w2-g128.pt     --save_path Experiments/opt-2.7b-w2-g128/qsd.pt >> Experiments/opt-2.7b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_LM.py     --model-name facebook/opt-2.7b     --load Experiments/opt-2.7b-w2-g128/qsd.pt     --output-dir Experiments/opt-2.7b-w2-g128/ 
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-2.7b     --load Experiments/opt-2.7b-w2-g128/qsd.pt     --output-dir Experiments/opt-2.7b-w2-g128/

CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path facebook/./llama2-hf/llama-2-13b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-13b-w4-g128.pt     --save_path Experiments/llama-2-13b-w4-g128/qsd.pt >> Experiments/llama-2-13b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w4-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w4-g128 
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w4-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w4-g128
CUDA_VISIBLE_DEVICES=1 python eval-nlp/eval_MMLU_oi.py     --model_name_or_path ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w4-g128/qsd.pt     --save_dir Experiments/llama-2-13b-w4-g128/MMLU_OI     --eval_batch_size 1

CUDA_VISIBLE_DEVICES=2 python -m awq.entry --model_path facebook/opt-1.3b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-1.3b-w4-g128.pt     --save_path Experiments/opt-1.3b-w4-g128/qsd.pt >> Experiments/opt-1.3b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_LM.py     --model-name facebook/opt-1.3b     --load Experiments/opt-1.3b-w4-g128/qsd.pt     --output-dir Experiments/opt-1.3b-w4-g128/ 
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-1.3b     --load Experiments/opt-1.3b-w4-g128/qsd.pt     --output-dir Experiments/opt-1.3b-w4-g128/

CUDA_VISIBLE_DEVICES=2 python -m awq.entry --model_path facebook/opt-2.7b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-2.7b-w3-g128.pt     --save_path Experiments/opt-2.7b-w3-g128/qsd.pt >> Experiments/opt-2.7b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_LM.py     --model-name facebook/opt-2.7b     --load Experiments/opt-2.7b-w3-g128/qsd.pt     --output-dir Experiments/opt-2.7b-w3-g128/ 
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-2.7b     --load Experiments/opt-2.7b-w3-g128/qsd.pt     --output-dir Experiments/opt-2.7b-w3-g128/

CUDA_VISIBLE_DEVICES=2 python -m awq.entry --model_path facebook/opt-6.7b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-6.7b-w2-g128.pt     --save_path Experiments/opt-6.7b-w2-g128/qsd.pt >> Experiments/opt-6.7b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_LM.py     --model-name facebook/opt-6.7b     --load Experiments/opt-6.7b-w2-g128/qsd.pt     --output-dir Experiments/opt-6.7b-w2-g128/ 
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-6.7b     --load Experiments/opt-6.7b-w2-g128/qsd.pt     --output-dir Experiments/opt-6.7b-w2-g128/

CUDA_VISIBLE_DEVICES=2 python -m awq.entry --model_path facebook/./llama2-hf/llama-2-7b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-7b-w3-g128.pt     --save_path Experiments/llama-2-7b-w3-g128/qsd.pt >> Experiments/llama-2-7b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w3-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w3-g128 
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w3-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w3-g128
CUDA_VISIBLE_DEVICES=2 python eval-nlp/eval_MMLU_oi.py     --model_name_or_path ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w3-g128/qsd.pt     --save_dir Experiments/llama-2-7b-w3-g128/MMLU_OI     --eval_batch_size 1

CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path facebook/opt-2.7b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-2.7b-w4-g128.pt     --save_path Experiments/opt-2.7b-w4-g128/qsd.pt >> Experiments/opt-2.7b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_LM.py     --model-name facebook/opt-2.7b     --load Experiments/opt-2.7b-w4-g128/qsd.pt     --output-dir Experiments/opt-2.7b-w4-g128/ 
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-2.7b     --load Experiments/opt-2.7b-w4-g128/qsd.pt     --output-dir Experiments/opt-2.7b-w4-g128/

CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path facebook/opt-6.7b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-6.7b-w3-g128.pt     --save_path Experiments/opt-6.7b-w3-g128/qsd.pt >> Experiments/opt-6.7b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_LM.py     --model-name facebook/opt-6.7b     --load Experiments/opt-6.7b-w3-g128/qsd.pt     --output-dir Experiments/opt-6.7b-w3-g128/ 
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-6.7b     --load Experiments/opt-6.7b-w3-g128/qsd.pt     --output-dir Experiments/opt-6.7b-w3-g128/

CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path facebook/opt-13b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-13b-w2-g128.pt     --save_path Experiments/opt-13b-w2-g128/qsd.pt >> Experiments/opt-13b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_LM.py     --model-name facebook/opt-13b     --load Experiments/opt-13b-w2-g128/qsd.pt     --output-dir Experiments/opt-13b-w2-g128/ 
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-13b     --load Experiments/opt-13b-w2-g128/qsd.pt     --output-dir Experiments/opt-13b-w2-g128/

CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path facebook/./llama2-hf/llama-2-13b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-13b-w3-g128.pt     --save_path Experiments/llama-2-13b-w3-g128/qsd.pt >> Experiments/llama-2-13b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w3-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w3-g128 
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w3-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w3-g128
CUDA_VISIBLE_DEVICES=3 python eval-nlp/eval_MMLU_oi.py     --model_name_or_path ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w3-g128/qsd.pt     --save_dir Experiments/llama-2-13b-w3-g128/MMLU_OI     --eval_batch_size 1

CUDA_VISIBLE_DEVICES=4 python -m awq.entry --model_path facebook/opt-6.7b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-6.7b-w4-g128.pt     --save_path Experiments/opt-6.7b-w4-g128/qsd.pt >> Experiments/opt-6.7b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_LM.py     --model-name facebook/opt-6.7b     --load Experiments/opt-6.7b-w4-g128/qsd.pt     --output-dir Experiments/opt-6.7b-w4-g128/ 
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-6.7b     --load Experiments/opt-6.7b-w4-g128/qsd.pt     --output-dir Experiments/opt-6.7b-w4-g128/

CUDA_VISIBLE_DEVICES=4 python -m awq.entry --model_path facebook/opt-13b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-13b-w3-g128.pt     --save_path Experiments/opt-13b-w3-g128/qsd.pt >> Experiments/opt-13b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_LM.py     --model-name facebook/opt-13b     --load Experiments/opt-13b-w3-g128/qsd.pt     --output-dir Experiments/opt-13b-w3-g128/ 
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-13b     --load Experiments/opt-13b-w3-g128/qsd.pt     --output-dir Experiments/opt-13b-w3-g128/

CUDA_VISIBLE_DEVICES=4 python -m awq.entry --model_path facebook/opt-30b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-30b-w2-g128.pt     --save_path Experiments/opt-30b-w2-g128/qsd.pt >> Experiments/opt-30b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_LM.py     --model-name facebook/opt-30b     --load Experiments/opt-30b-w2-g128/qsd.pt     --output-dir Experiments/opt-30b-w2-g128/ 
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-30b     --load Experiments/opt-30b-w2-g128/qsd.pt     --output-dir Experiments/opt-30b-w2-g128/

CUDA_VISIBLE_DEVICES=4 python -m awq.entry --model_path facebook/./llama2-hf/llama-2-7b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-7b-w2-g128.pt     --save_path Experiments/llama-2-7b-w2-g128/qsd.pt >> Experiments/llama-2-7b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w2-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w2-g128 
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w2-g128/qsd.pt     --output-dir Experiments/llama-2-7b-w2-g128
CUDA_VISIBLE_DEVICES=4 python eval-nlp/eval_MMLU_oi.py     --model_name_or_path ./llama2-hf/llama-2-7b     --load Experiments/llama-2-7b-w2-g128/qsd.pt     --save_dir Experiments/llama-2-7b-w2-g128/MMLU_OI     --eval_batch_size 1

CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path facebook/opt-13b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-13b-w4-g128.pt     --save_path Experiments/opt-13b-w4-g128/qsd.pt >> Experiments/opt-13b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_LM.py     --model-name facebook/opt-13b     --load Experiments/opt-13b-w4-g128/qsd.pt     --output-dir Experiments/opt-13b-w4-g128/ 
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-13b     --load Experiments/opt-13b-w4-g128/qsd.pt     --output-dir Experiments/opt-13b-w4-g128/

CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path facebook/opt-30b     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-30b-w3-g128.pt     --save_path Experiments/opt-30b-w3-g128/qsd.pt >> Experiments/opt-30b-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_LM.py     --model-name facebook/opt-30b     --load Experiments/opt-30b-w3-g128/qsd.pt     --output-dir Experiments/opt-30b-w3-g128/ 
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-30b     --load Experiments/opt-30b-w3-g128/qsd.pt     --output-dir Experiments/opt-30b-w3-g128/

CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path facebook/./llama2-hf/llama-2-13b     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/./llama2-hf/llama-2-13b-w2-g128.pt     --save_path Experiments/llama-2-13b-w2-g128/qsd.pt >> Experiments/llama-2-13b-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_LM.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w2-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w2-g128 
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_lm_eval.py     --model-name ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w2-g128/qsd.pt     --output-dir Experiments/llama-2-13b-w2-g128
CUDA_VISIBLE_DEVICES=5 python eval-nlp/eval_MMLU_oi.py     --model_name_or_path ./llama2-hf/llama-2-13b     --load Experiments/llama-2-13b-w2-g128/qsd.pt     --save_dir Experiments/llama-2-13b-w2-g128/MMLU_OI     --eval_batch_size 1

CUDA_VISIBLE_DEVICES=6 python -m awq.entry --model_path facebook/opt-30b     --w_bit 4     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-30b-w4-g128.pt     --save_path Experiments/opt-30b-w4-g128/qsd.pt >> Experiments/opt-30b-w4-g128/log.txt
CUDA_VISIBLE_DEVICES=6 python eval-nlp/eval_LM.py     --model-name facebook/opt-30b     --load Experiments/opt-30b-w4-g128/qsd.pt     --output-dir Experiments/opt-30b-w4-g128/ 
CUDA_VISIBLE_DEVICES=6 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-30b     --load Experiments/opt-30b-w4-g128/qsd.pt     --output-dir Experiments/opt-30b-w4-g128/

CUDA_VISIBLE_DEVICES=6 python -m awq.entry --model_path facebook/opt-125m     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-125m-w2-g128.pt     --save_path Experiments/opt-125m-w2-g128/qsd.pt >> Experiments/opt-125m-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=6 python eval-nlp/eval_LM.py     --model-name facebook/opt-125m     --load Experiments/opt-125m-w2-g128/qsd.pt     --output-dir Experiments/opt-125m-w2-g128/ 
CUDA_VISIBLE_DEVICES=6 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-125m     --load Experiments/opt-125m-w2-g128/qsd.pt     --output-dir Experiments/opt-125m-w2-g128/

CUDA_VISIBLE_DEVICES=7 python -m awq.entry --model_path facebook/opt-125m     --w_bit 3     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-125m-w3-g128.pt     --save_path Experiments/opt-125m-w3-g128/qsd.pt >> Experiments/opt-125m-w3-g128/log.txt
CUDA_VISIBLE_DEVICES=7 python eval-nlp/eval_LM.py     --model-name facebook/opt-125m     --load Experiments/opt-125m-w3-g128/qsd.pt     --output-dir Experiments/opt-125m-w3-g128/ 
CUDA_VISIBLE_DEVICES=7 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-125m     --load Experiments/opt-125m-w3-g128/qsd.pt     --output-dir Experiments/opt-125m-w3-g128/

CUDA_VISIBLE_DEVICES=7 python -m awq.entry --model_path facebook/opt-350m     --w_bit 2     --q_group_size 128 --run_awq     --dump_awq awq_cache_run/opt-350m-w2-g128.pt     --save_path Experiments/opt-350m-w2-g128/qsd.pt >> Experiments/opt-350m-w2-g128/log.txt
CUDA_VISIBLE_DEVICES=7 python eval-nlp/eval_LM.py     --model-name facebook/opt-350m     --load Experiments/opt-350m-w2-g128/qsd.pt     --output-dir Experiments/opt-350m-w2-g128/ 
CUDA_VISIBLE_DEVICES=7 python eval-nlp/eval_lm_eval.py     --model-name facebook/opt-350m     --load Experiments/opt-350m-w2-g128/qsd.pt     --output-dir Experiments/opt-350m-w2-g128/