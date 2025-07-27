# 第二轮考察

> 姓名：董博伟      本科学校：中国海洋大学      日期：2025年7月27日

### 选择主题：

**09-轻量化与高效部署——(Knowledge Distillation)知识蒸馏**

### 选择论文：

***Decoupled Knowledge Distillation***

```
@article{zhao2022dkd,
  title={Decoupled Knowledge Distillation},
  author={Zhao, Borui and Cui, Quan and Song, Renjie and Qiu, Yiyu and Liang, Jiajun},
  journal={arXiv preprint arXiv:2203.08679},
  year={2022}
}
```

### 任务介绍：

论文《Decoupled Knowledge Distillation》致力于解决传统知识蒸馏（KD）方法在logit蒸馏方面效果不佳的问题。以往的研究更关注中间层特征的蒸馏，尽管这类方法性能较好，却伴随着高昂的计算和存储成本。作者认为logit本身作为高语义层次的输出应具备较强的信息表达能力，因此提出重新审视logit蒸馏的潜力。

该研究的核心创新是将经典的KD损失函数解耦为两部分：目标类知识蒸馏（TCKD）和非目标类知识蒸馏（NCKD）。TCKD传递关于训练样本“难度”的知识，而NCKD则保留了学生对非目标类的判别能力，实验证明NCKD是logit蒸馏中最有效的部分。然而，传统KD中这两部分是耦合的，特别是NCKD的权重与教师模型对目标类的置信度负相关，从而导致对预测准确样本的知识传递被抑制。

为此，作者提出了DKD方法，通过引入两个超参数分别控制TCKD和NCKD的权重，彻底解耦二者。这种方式不仅提升了蒸馏灵活性，也增强了对高置信度样本的知识利用能力。

该论文开源代码基于Pytorch架构实现，实验在CIFAR-100、ImageNet和MS-COCO等多个任务上进行，DKD在训练效率、性能表现及特征迁移能力方面均优于传统KD方法，部分情况下甚至超过了特征蒸馏方法，显示出较强的实用价值和研究前景。

本任务基于原论文所提方法及其开源代码，使用Jittor架构实现。考虑到计算资源和训练时间有限，本任务使用**CIFAR-100数据集**进行实验。模型超参数，优化器，训练设置等信息均与原文一致，硬件设备方面，原文仅提及在CIFAR-100数据集上的实验使用单卡训练和测试，未详细说明硬件型号，本任务使用**环境及硬件**如下：

```
环境：jittor-1.3.1；Python3.8；CUDA11.3；ubuntu18.04
GPU：RTX 3090 E5 * 1
CPU：Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz * 7核
```

在实验设计方面，本任务分别使用原始Pytorch架构代码和本任务实现的Jittor架构代码在图像分类数据集CIFAR-100数据集上对**11组teacher-student模型**进行测试，包括6组相同架构模型和5组不同架构模型，详细信息如下：

​	相同架构模型：

- ResNet56-ResNet20
- ResNet110-ResNet32
- ResNet32×4-ResNet8×4
- WRN-40-2-WRN-16-2
- WRN-40-2-WRN-40-1
- VGG13-VGG8

​	不同架构模型：

- ResNet32×4-ShuffleNet-V1
- WRN-40-2-ShuffleNet-V1
- VGG13-MobileNet-V2
- ResNet50-MobileNet-V2
- ResNet32×4-ShuffleNet-V2

实验内容包括：

1. **相同架构模型在CIFAR-100 验证集上的测试精度对比**
2. **不同架构模型在CIFAR-100 验证集上的测试精度对比**
3. **相同架构模型在CIFAR-100 验证集上的推理速度对比**
4. **不同架构模型在CIFAR-100 验证集上的推理速度对比**

### 环境配置脚本：

运行以下指令配置环境：

```
sudo pip install -r requirements.txt
sudo python setup.py develop
```

### 数据准备脚本：

请访问 https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 获取CIFAR-100数据集，解压并将检查点下载至 `./data` 目录。

### 训练脚本：

请访问 https://github.com/megvii-research/mdistiller/releases/tag/checkpoints 获取`cifar_teachers.tar`文件，解压并将检查点下载至 `./download_ckpts` 目录。

分别运行以下命令训练11组不同架构组合的teacher-student模型：

```
python tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml
python tools/train.py --cfg configs/cifar100/dkd/res110_res32.yaml
python tools/train.py --cfg configs/cifar100/dkd/res56_res20.yaml
python tools/train.py --cfg configs/cifar100/dkd/vgg13_vgg8.yaml
python tools/train.py --cfg configs/cifar100/dkd/wrn40_2_wrn_16_2.yaml
python tools/train.py --cfg configs/cifar100/dkd/wrn40_2_wrn_40_1.yaml
python tools/train.py --cfg configs/cifar100/dkd/res32x4_shuv1.yaml
python tools/train.py --cfg configs/cifar100/dkd/wrn40_2_shuv1.yaml
python tools/train.py --cfg configs/cifar100/dkd/vgg13_mv2.yaml
python tools/train.py --cfg configs/cifar100/dkd/res50_mv2.yaml
python tools/train.py --cfg configs/cifar100/dkd/res32x4_shuv2.yaml
```

### 测试脚本：

请访问 https://github.com/megvii-research/mdistiller/releases/tag/checkpoints 获取`dkd_resnet8x4`文件，请解压并将检查点下载至 `./download_ckpts` 目录。

测试教师模型：

```
python tools/eval.py -m resnet32x4
```

测试预训练学生模型：

```
python tools/eval.p -m resnet8x4 -c download_ckpts/dkd_resnet8x4
```

分别运行以下模型测试本任务训练的学生模型：

```
python tools/eval.py -m resnet8x4 -c output_0/cifar100_baselines/dkd,res32x4,res8x4/student_best
python tools/eval.py -m resnet32 -c output_1/cifar100_baselines/dkd,res110,res32/student_best
python tools/eval.py -m resnet20 -c output_2/cifar100_baselines/dkd,res56,res20/student_best
python tools/eval.py -m vgg8 -c output_3/cifar100_baselines/dkd,vgg13,vgg8/student_best
python tools/eval.py -m wrn_16_2 -c output_4/cifar100_baselines/dkd,wrn_40_2,wrn_16_2/student_best
python tools/eval.py -m wrn_40_1 -c output_5/cifar100_baselines/dkd,wrn_40_2,wrn_40_1/student_best
python tools/eval.py -m ShuffleV1 -c output_6/cifar100_baselines/dkd,res32x4,shuv1/student_best
python tools/eval.py -m ShuffleV1 -c output_7/cifar100_baselines/dkd,wrn_40_2,shuv1/student_best
python tools/eval.py -m MobileNetV2 -c output_8/cifar100_baselines/dkd,vgg13,mv2/student_best
python tools/eval.py -m MobileNetV2 -c output_9/cifar100_baselines/dkd,res50,mv2/student_best
python tools/eval.py -m ShuffleV2 -c output_10/cifar100_baselines/dkd,res32x4,shuv2/student_best
```

### 实验结果：

以下为Pytorch架构和Jittor架构测试精度对齐实验和推理速度对齐实验的实验结果：

![表1](.\README_Image\表1.png)

**说明：表中Paper行数值为原论文提供。△行绿色数值为同比增长项。Fluctuation行数值为|△÷Pytorch|%，表示Jittor架构相对于Pytorch架构的震荡，均在0.5%以内，属于可接受范围。**

![表2](.\README_Image\表2.png)

**说明：表中Paper行数值为原论文提供。△行绿色数值为同比增长项。Fluctuation行数值为|△÷Pytorch|%，表示Jittor架构相对于Pytorch架构的震荡，基本在1%以内，属于可接受范围。**

![表3](.\README_Image\表3.png)

**说明：表中Mean_Pytorch行数值为Pytorch行三次测试结果的平均值，Mean_Jittor同理。△行红色数值为Jittor架构相比于Pytorch架构的推理速度减少时间。Growth行数值为(△÷Pytorch)%，使用Jittor实现的相同架构模型的推理速度同比增长30%以上，显著提高模型推理速度。**

![表4](.\README_Image\表4.png)

**说明：表中Mean_Pytorch行数值为Pytorch行三次测试结果的平均值，Mean_Jittor同理。△行红色数值为Jittor架构相比于Pytorch架构的推理速度减少时间。Growth行数值为(△÷Pytorch)%，使用Jittor实现的相同架构模型的推理速度同比增长30%以上，显著提高模型推理速度。**

### 训练过程：

模型ResNet32×4-ResNet8×4在Jittor架构下的训练log部分示例如下，其余模型Pytorch架构和Jittor架构训练对齐log分别位于文件夹mdistiller-master-jittor/worklog中。

```
-------------------------
epoch: 1
lr: 0.05
train_acc: 12.45
train_loss: 4.83
test_acc: 18.29
test_acc_top5: 46.42
test_loss: 3.33
-------------------------
-------------------------
epoch: 2
lr: 0.05
train_acc: 25.58
train_loss: 4.93
test_acc: 27.10
test_acc_top5: 58.25
test_loss: 2.89
-------------------------
...
-------------------------
epoch: 239
lr: 0.00
train_acc: 89.99
train_loss: 6.31
test_acc: 75.81
test_acc_top5: 94.14
test_loss: 0.91
-------------------------
-------------------------
epoch: 240
lr: 0.00
train_acc: 90.05
train_loss: 6.30
test_acc: 75.79
test_acc_top5: 94.05
test_loss: 0.91
-------------------------
best_acc	75.90
```

所有模型完整训练结果（包括教师模型，学生模型，不同epoch保存的检查点等内容）分别位于：mdistiller-master-jittor/output_i文件夹下，其中i为从0到10的所有，分别对应11组模型，具体对应模型以mdistiller-master-jittor/output_i/cifar100_baselines目录下文件夹名称为准。

### Loss：

以下为Pytorch架构和Jittor架构的训练损失曲线和精度曲线：

模型ResNet32×4-ResNet8×4的Pytorch架构和Jittor架构训练损失曲线和精度曲线对齐实验：

![output1](.\README_Image\output1.png)

模型ResNet32×4-ShuffleNet-V1的Pytorch架构和Jittor架构训练损失曲线和精度曲线对齐实验：

![output2](.\README_Image\output2.png)

其余模型在Pytorch架构和Jittor架构下的训练损失曲线和精度曲线对齐结果分别位于文件夹mdistiller-master-jittor/loss中。

### 文件组织：

```
mdistiller-master-jittor
    ├─configs
    │  ├─cifar100
    │  │  ├─dkd
    │  │  └─dot
    │  ├─imagenet
    │  │  ├─r34_r18
    │  │  └─r50_mv1
    │  └─tiny_imagenet
    │      └─dot
    ├─data
    │  └─cifar-100-python
    ├─detection
    │  ├─configs
    │  │  ├─DKD
    │  │  └─ReviewKD
    │  └─model
    │      ├─backbone
    │      └─teacher
    ├─download_ckpts
    │  └─cifar_teachers
    ├─mdistiller
    │  ├─dataset
    │  ├─distillers
    │  ├─engine
    │  └─models
    │      ├─cifar
    │      └─imagenet
    ├─output
    │  └─cifar100_baselines
    │      └─dkd,res32x4,res8x4
    ├─loss
    │  ├─Jittor
    │  └─Pytorch
    ├─worklog
    │  ├─Jittor
    │  └─Pytorch
    ├─setup.py
    ├─README.md
    ├─requirements.txt
    └─tools
	   ├─train.py
       ├─eval.py
	   └─visualizations
```

### 致谢：

- 感谢 DKD。我基于 [DKD 的代码库](https://github.com/megvii-research/mdistiller) 完成了该任务。

