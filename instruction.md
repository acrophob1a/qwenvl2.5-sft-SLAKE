基于Qwen2.5-VL的视觉语言问答(VQA)微调
项目简介
本项目主要基于 Qwen2.5-VL 模型，在自定义选择的数据集上进行监督微调（SFT, Supervised Fine-Tuning）。

核心涵盖内容：

Qwen-VL系列模型代码

VQA数据集的预处理过程（包括 dataloader、data collator）

代码下载、环境配置、数据处理、训练预测流程。

项目在简历上的呈现技巧及可能会被问到的面试问题。

项目教程
1. AutoDL实例资源选择
由于多模态大模型的显存需求较大，在配置机器时请参考以下建议：

显存要求： 尽量选择 80G 显存的显卡（如 A800）。

显卡数量： 数量选择任意，单卡性能已足够完成本项目。

框架版本： PyTorch (torch) 版本请选择 2.6.0。

2. 如何下载代码仓库
自行设定你的本地目录 ${your_local_dir}，下载后进入主目录进行解压。

Bash
hf download Brilliant-B/awesome-demos demo1.tar.gz --local-dir ${your_local_dir} 

# 解压到指定的路径 
tar -xvf ${your_local_dir}/demo1.tar.gz -C /root/autodl-tmp/${your_root}
3. 运行环境配置
在终端中依次运行以下命令以安装所需依赖：

Bash
pip install torchvision==0.21.0 

# 进入到主目录，安装依赖库 
pip install -v -e . 
pip install -v -e finetuning
4. 基座模型下载
目标模型： Qwen2.5-VL-3B-Instruct

操作指引： 建议创建 pretrained 目录，设置镜像地址后完成模型的下载。