#!/bin/bash 

python xp_train.py -F './runs' with\
       model_id="camembert-base"\
       output_dir="./models"\
       hg_training_kwargs.output_dir="./checkpoints"\
       hg_training_kwargs.overwrite_output_dir=True\
       hg_training_kwargs.per_device_train_batch_size=8\
       hg_training_kwargs.per_device_eval_batch_size=8\
       hg_training_kwargs.num_train_epochs=3\
       hg_training_kwargs.save_total_limit=3\
       hg_training_kwargs.eval_accumulation_steps=32\
       hg_training_kwargs.eval_strategy="epoch"\
       hg_training_kwargs.learning_rate=1e-5\
       use_weights=True
