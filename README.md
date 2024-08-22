## Probabilistically Robust Watermarking of Neural Networks.
This is the official repository of the [IJCAI-2024 accepted paper](https://www.ijcai.org/proceedings/2024/528).

## Abstract.
As deep learning (DL) models are widely and effectively used in Machine Learning as a Service (MLaaS) platforms, there is a rapidly growing interest in DL watermarking techniques that can be used to confirm the ownership of a particular model. Unfortunately, these methods usually produce watermarks susceptible to model stealing attacks. In our research, we introduce a novel trigger set-based watermarking approach that demonstrates resilience against functionality stealing attacks, particularly those involving extraction and distillation. Our approach does not require additional model training and can be applied to any model architecture. The key idea of our method is to compute the trigger set, which is transferable between the source model and the set of proxy models with a high probability. In our experimental study, we show that if the probability of the set being transferable is reasonably high, it can be effectively used for ownership verification of the stolen model. We evaluate our method on multiple benchmarks and show that our approach outperforms current state-of-the-art watermarking techniques in all considered experimental setups.

## Commands.
Create stealing models:

```python distillation_extraction.py --dataset cifar10  --model_name resnet34 --model_path ./teacher_cifar10_resnet34/model_1 --student_name vgg11 --policy soft --num_models 10```

Create watermarking and test on saved models:

```python create_test_watermarks.py --dataset cifar10 --model_name resnet34 --model_path ./teacher_cifar10_resnet34/model_1 --student_name vgg11 --student_path ./stealing_vgg11_cifar10_soft --sigma1 8e-3 --M 64 --N 100```

Available settings:
1. You can use train split instead of test by changing ``--use_train`` to True.
2. You can control the maximum accuracy deviation of proxy models using a "threshold".

## Citing.
If you use this package in your publications or in other work, please cite it as follows:
```
@inproceedings{PROWN,
  title     = {Probabilistically Robust Watermarking of Neural Networks},
  author    = {Pautov, Mikhail and Bogdanov, Nikita and Pyatkin, Stanislav and Rogov, Oleg and Oseledets, Ivan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {4778--4787},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/528},
  url       = {https://doi.org/10.24963/ijcai.2024/528},
}
```
