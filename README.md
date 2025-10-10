# Euclid Machine

Automatic Construction of Plane Geometry System without Human Supervision.

## 环境配置

自行下载并学习使用相关工具，如版本控制工具Git 、Python IDE Pycharm、远程终端管理软件MobaXterm、环境管理软件Conda等。

下面`$`表示Git Bash执行，`>`表示系统命令行执行，根目录用`Projects/`表示。

### 下载项目

在根目录下右键，选择Open Git Bash here，打开Git Bash，从远程仓库下载项目：

    Projects/ $ git clone https://github.com/BitSecret/EuclidMachine.git

### Python环境配置

打开系统命令行，进入项目目录，使用Conda新建Python环境，并安装依赖：

    Projects/ > cd EuclidMachine
    Projects/EuclidMachine/ > conda create -n EuclidMachine python=3.10.18
    Projects/EuclidMachine/ > conda activate EuclidMachine
    (EuclidMachine) Projects/EuclidMachine/ > pip install -e .

### 深度学习环境配置

安装最近Nvidia GPU驱动，以及CUDA Toolkit (>=12.1)，之后下载安装深度学习库Pytorch：

    (EuclidMachine) Projects/EuclidMachine/ > conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia

### 新建data目录

生成的数据不会上传到仓库，每个人需要在其本地新建data目录，目录结构为：

    EuclidMachine/
    ├── data/
    │   ├── imo sl/
    │   │   ├── 1.json
    │   │   ├── 2.josn
    │   │   ├── ...
    │   │   └── n.json
    │   └── tips/
    │       ├── API文档笔记.txt
    │       └── notes.py
    ├── src/
    │   └── em/
    │       ├── constructor/
    │       │    ├── 1.py
    │       │    └── 2.py
    │       ├── formalgeo/
    │       │    ├── 1.py
    │       │    └── 2.py
    │       ├── inductor/
    │       │    ├── 1.py
    │       │    └── 2.py
    │       └── solver/
    │            ├── 1.py
    │            └── 2.py
    ├── .gitignore
    ├── LICENSE
    ├── pyproject.toml
    └── README.md

## 协作

我们使用Git协作，并使用Github作为远程协作仓库。在项目目录下右键，选择Open Git Bash here，打开Git Bash。

### 首次

新建你自己的个人分支`your_name`（如`xiaokai`）并推送到远程仓库：

    Projects/EuclidMachine/ $ git checkout -b your_name
    Projects/EuclidMachine/ $ git push --set-upstream origin your_name

### 开发过程

在每次提交代码前，跟主分支同步：

    Projects/EuclidMachine/ $ git pull origin main

提交代码，并推送到远程仓库。注意不要修改你负责范围（例如我的是`src/em/formalgeo`）之外的代码，否则会引起冲突。
如果需要更改其他部分代码，让张效凯直接在主分支更改。

    Projects/EuclidMachine/ $ git add src/em/formalgeo
    Projects/EuclidMachine/ $ git commit -m "briefly describe this commit"
    Projects/EuclidMachine/ $ git push