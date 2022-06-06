## Installation

```bash
pip install -r requirements.txt
```

## File Structure

```
.
+-- models/ (Folder containing all model definitions)
|   +-- resnet.py (containing the ResNet18 model)
|   +-- vgg.py (containing the VGG16 model)
|   +-- preactresnet.py (containing the PreActResNet18 model)
|   +-- utils.py (utility functions and modules)
+-- datasets.py (containing functions to load data)
+-- train_node_bnn.py (script for training node BNNs)
+-- train_sgd.py (script for training deterministic models)
```

## Command to replicate the result

Training VGG16 on CIFAR-10
```bash
srun python train_node_bnn.py.py with model_name=StoVGG16 validation=False validation_fraction=5000 augment_data=True "kl_type=upper_bound" "gamma=<GAMMA>" entropy_type=BD \
                          num_epochs=300 save_freq=301 logging_freq=1 'kl_weight.kl_min=0.0' "kl_weight.kl_max=1.0" 'kl_weight.last_iter=200' lr_ratio_det=0.01 lr_ratio_sto=1.0 \
                          prior_std=0.30 prior_mean=1.0 "det_params.weight_decay=0.0005" n_components=4 dataset=cifar10 "posterior_mean_init=(1.0,0.05)" "posterior_std_init=(0.30,0.02)" \
                          "det_params.lr=0.05" 'sto_params.lr'=0.05 'sto_params.weight_decay=0.0' "sto_params.momentum=0.0" 'sto_params.nesterov=True' 'num_train_sample'=4 bn_momentum=0.1 \
                          noise_mode=out 'sgd_params.nesterov'=True 'det_milestones=(0.50,0.90)' \
                          name=<UNIQUE_NAME_FOR_THE_EXPERIMENT> \
                          batch_size=128 test_batch_size=512 seed=<RANDOM_SEED> num_test_sample=8
```
Training VGG16 on CIFAR-100
```bash
srun python train_node_bnn.py.py with model_name=StoVGG16 validation=False validation_fraction=5000 augment_data=True "kl_type=upper_bound" "gamma=<GAMMA>" entropy_type=BD \
                          num_epochs=300 save_freq=301 logging_freq=1 'kl_weight.kl_min=0.0' "kl_weight.kl_max=1.0" 'kl_weight.last_iter=200' lr_ratio_det=0.01 lr_ratio_sto=1.0 \
                          prior_std=0.30 prior_mean=1.0 "det_params.weight_decay=0.0005" n_components=4 dataset=cifar100 "posterior_mean_init=(1.0,0.05)" "posterior_std_init=(0.30,0.02)" \
                          "det_params.lr=0.05" 'sto_params.lr'=0.05 'sto_params.weight_decay=0.0' "sto_params.momentum=0.0" 'sto_params.nesterov=True' 'num_train_sample'=4 bn_momentum=0.1 \
                          noise_mode=out 'sgd_params.nesterov'=True 'det_milestones=(0.50,0.90)' \
                          name=<UNIQUE_NAME_FOR_THE_EXPERIMENT> \
                          batch_size=128 test_batch_size=512 seed=<RANDOM_SEED> num_test_sample=8
```
Training ResNet18 on CIFAR-10
```bash
srun python train_node_bnn.py.py with model_name=StoResNet18 validation=False validation_fraction=5000 augment_data=True "kl_type=upper_bound" "gamma=<GAMMA>" entropy_type=BD \
                          num_epochs=300 save_freq=301 logging_freq=1 'kl_weight.kl_min=0.0' "kl_weight.kl_max=1.0" 'kl_weight.last_iter=200' lr_ratio_det=0.01 lr_ratio_sto=1.0 \
                          prior_std=0.40 prior_mean=1.0 "det_params.weight_decay=0.0005" n_components=4 dataset=cifar10 "posterior_mean_init=(1.0,0.05)" "posterior_std_init=(0.40,0.02)" \
                          "det_params.lr=0.10" 'sto_params.lr'=0.10 'sto_params.weight_decay=0.0' "sto_params.momentum=0.0" 'sto_params.nesterov=True' 'num_train_sample'=4 bn_momentum=0.1 \
                          noise_mode=out 'sgd_params.nesterov'=True 'det_milestones=(0.50,0.90)' \
                          name=<UNIQUE_NAME_FOR_THE_EXPERIMENT> \
                          batch_size=128 test_batch_size=512 seed=<RANDOM_SEED> num_test_sample=8
```
Training ResNet18 on CIFAR-100
```bash
srun python train_node_bnn.py.py with model_name=StoResNet18 validation=False validation_fraction=5000 augment_data=True "kl_type=upper_bound" "gamma=<GAMMA>" entropy_type=BD \
                          num_epochs=300 save_freq=301 logging_freq=1 'kl_weight.kl_min=0.0' "kl_weight.kl_max=1.0" 'kl_weight.last_iter=200' lr_ratio_det=0.01 lr_ratio_sto=1.0 \
                          prior_std=0.40 prior_mean=1.0 "det_params.weight_decay=0.0005" n_components=4 dataset=cifar100 "posterior_mean_init=(1.0,0.05)" "posterior_std_init=(0.40,0.02)" \
                          "det_params.lr=0.10" 'sto_params.lr'=0.10 'sto_params.weight_decay=0.0' "sto_params.momentum=0.0" 'sto_params.nesterov=True' 'num_train_sample'=4 bn_momentum=0.1 \
                          noise_mode=out 'sgd_params.nesterov'=True 'det_milestones=(0.50,0.90)' \
                          name=<UNIQUE_NAME_FOR_THE_EXPERIMENT> \
                          batch_size=128 test_batch_size=512 seed=<RANDOM_SEED> num_test_sample=8
```
For more information on each training option, please read the comments in the `train.py` file.
Each experiment will be stored in a subfolder of the `experiments` folder.
