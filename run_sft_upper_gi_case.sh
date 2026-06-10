# --------- VLM + LLM（该思路参数一致，需将数据集路径更换） --------------
# --train_file_dir /media/inno/VLM/D1_images_and_reports_which_have_video_20250316/MedicalGPT/V3/train/ \
# --validation_file_dir /media/inno/VLM/D1_images_and_reports_which_have_video_20250316/MedicalGPT/V3/test/ \

# ---------- ASR + LLM （ 其LLM采用病例描述，即术后一段话描述） --------------
python supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --train_file_dir /media/inno/LLM/report/V5/train/ \
    --validation_file_dir /media/inno/LLM/report/V5/test/ \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --model_max_length 4096 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --warmup_steps 5 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy epoch \
    --save_steps 500 \
    --save_strategy epoch \
    --save_total_limit 13 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir /media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16-epochs5 \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache --flash_attn True
