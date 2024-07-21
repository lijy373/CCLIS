echo 'training seed 0 start'
python CCLIS/main.py --batch_size 512 --dataset cifar10 --model resnet18 --mem_size 200 --epochs 100 --start_epoch 500 --learning_rate 1.0 --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --cosine --syncBN --seed 0

wait

echo 'testing seed 0 start'
python CCLIS/main_linear_buffer.py --dataset cifar10  --learning_rate 0.05 --target_task 4 --ckpt ./save_weight_200/cifar10_models/cifar10_32_resnet18_lr_1.0_0.01_decay_0.0001_bsz_512_temp_0.5_trial_0_500_100_0.2_0.1_0.6_distill_type_PRD_freeze_prototypes_niters_5_seed_0_cosine_warm/ --logpt ./save_weight_200/logs/cifar10_32_resnet18_lr_1.0_0.01_decay_0.0001_bsz_512_temp_0.5_trial_0_500_100_0.2_0.1_0.6_distill_type_PRD_freeze_prototypes_niters_5_seed_0_cosine_warm/