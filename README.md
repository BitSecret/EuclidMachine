# Euclid Machine

Automatic Construction of Plane Geometry System without Human Supervision.

## 环境配置与协作

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

### 新建个人分支

我们使用Git协作，并使用Github作为远程协作仓库。在项目目录下右键，选择Open Git Bash here，打开Git Bash。

新建你自己的个人分支`your_name`（如`xiaokai`）并推送到远程仓库：

    Projects/EuclidMachine/ $ git checkout -b your_name
    Projects/EuclidMachine/ $ git push --set-upstream origin your_name

### 开发过程协作

在每次提交代码前，跟主分支同步：

    Projects/EuclidMachine/ $ git pull origin main

提交代码，并推送到远程仓库。注意不要修改你负责范围（例如我的是`src/em/formalgeo`）之外的代码，否则会引起冲突。
如果需要更改其他部分代码，让张效凯直接在主分支更改。

    Projects/EuclidMachine/ $ git add src/em/formalgeo
    Projects/EuclidMachine/ $ git commit -m "briefly describe this commit"
    Projects/EuclidMachine/ $ git push

## 几何定义语言设计

几何定义语言分为4部分，分别是实体（Entity）、测度（Measure）、关系（Relation）和公理（Axiom）。实体定义了基本的3种图形，分别是点（Point）、
线（Line）和圆（Circle）；测度是对于单个或多个实体某个测度的定义，分为参数（Parameter）和属性（Attribution），前者用于构图过程图形的约束满足
性判定和绘图，后者用于推理过程中代数关系的表示；关系是对于实体间几何关系的定义，同时定义了等价的实体参数间的代数约束，可用于构图过程；公理是系统最
基础最底层的推理规则，分为基础公理和扩展定理两类。

### 实体（Entity）

实体定义了基本的3种图形，分别是点（Point）、线（Line）和圆（Circle）。我们可以用单个拉丁字母或希腊字母（大小写均可）来表示一个基本图形，可使用的
字母库参考`em.formalgeo.tools`模块，如：

    Point(A), Line(l), Circle(Ω)

此外，我们在构图过程中可以使用临时形式来表示实体，并且**只能在构图过程中使用临时形式**。使用临时形式的实体最终会被解析为隐实体，隐实体的定义可以参
考对于关系的介绍这一小节。

点没有临时形式，线的临时形式有两种：

    Line(AB)  # 表示过A、B两点的直线
    Line(A;l)  # 表示过A点且斜率为l的直线

圆的临时形式也有两种：

    Circle(ABC)  # 表示过A、B、C三点的圆
    Circle(O;AB)  # 表示圆心为O点且半径与AB两点之间距离相等的圆

### 测度（Measure）

测度是对于单个或多个实体某个测度的定义，分为参数（Parameter）和属性（Attribution），前者用于构图过程图形的约束满足性判定和绘图，后者用于推理过
程中代数关系的表示。接下来分别给出实体参数和示例属性定义的例子。

#### 参数（Parameter）

实体参数的定义：

    "XCoordinateOfPoint(A)": {
      "type": "parameter",
      "ee_check": [
        "Point(A)"
      ],
      "sym": "x",
      "multi": []
    }

#### 属性（Attribution）

实体属性的定义：

    "DistanceBetweenPointAndPoint(A,B)": {
      "type": "attribution",
      "ee_check": [
        "Point(A)",
        "Point(B)"
      ],
      "sym": "dpp",
      "multi": [
        "DistanceBetweenPointAndPoint(B,A)"
      ]
    }

`ee_check`
用于实体存在性（或者说是实体依赖性）检查。在推理过程中，推导出来的结论只有在满足实体存在性的情况下，才会被添加到已知条件集`facts`
中，并记录下实体依赖关系，以供后续合成辅助作图数据。

`sym`定义了测度的符号表示。以上述两个定义为例，在代数关系中，点的X坐标表示为`A.x`，两点之间的距离表示为`AB.dpp`。

`multi`定义了测度符号表示的多种形式。以两点之间的距离为例，`DistanceBetweenPointAndPoint(A,B)`
与`DistanceBetweenPointAndPoint(B,A)`实际表示同一事实，所以在FormalGeo中会被解析为单一符号表示（AB.dpp或BA.dpp，看哪个先出现）。

### 关系（Relation）

关系是对于实体间几何关系的定义，同时定义了等价的实体参数间的代数约束，可用于构图过程。关系分为3类，分别是基础关系（Basic）、组合关系（Composite）
和间接关系（Indirect）。

#### 基础关系

基础关系是整个系统最核心的定义，组合关系和间接关系都由其扩展而来。一个基础关系的定义示例：

    "SamePoint(A,B)": {
      "type": "basic",
      "ee_check": [
        "Point(A)",
        "Point(B)"
      ],
      "extend": [
        "SamePoint(B,A)"
      ],
      "implicit_entity": [],
      "constraints": "Eq(Sub(A.x,B.x))&Eq(Sub(A.y,B.y))"
    }

`ee_check`用于实体存在性（或者说是实体依赖性）检查。在构图过程中，只有通过实体存在性检查的约束才能参与构图；在推理过程中，推导出来的结论只有在满
足实体存在性的情况下，才会被添加到已知条件集`facts`中，并记录下实体依赖关系，以供后续合成辅助作图数据。

`extend`定义了自动扩展过程，描述了当前关系的定义或常识。在FormalGeo中，添加关系后，会自动扩展出`extend`中定义的内容。以上述定义为例，当关系
`SamePoint(A,B)`添加到`facts`中时，`SamePoint(B,A)`会被自动扩展出来。

`implicit_entity`定义了间接关系依赖的实体，将在后续例子中详细解释。

`constraints`从代数角度定义了当前关系。以上述定义为例，*两个点相同*这个几何关系，可以用*两个点的坐标相等*
这个代数关系表示。代数关系使用代数表达式定义，大致包含4类元素：
1. 约束类型。5种约束类型分别为：等于0（Eq）、小于0（L）、小于等于0（Leq）、大于0（G）、大于等于0（Geq）、不等于0（Ueq）。使用时，将代
   数表达式置于约束内部，如`Eq(Sub(A.x,B.x))`。
2. 逻辑运算。 ~~3种逻辑运算分别为：且（&）、或（|）、非（~）。我们可以使用逻辑运算将各种约束组合起来，如"Eq(expr)&(Leq(expr)|Geq(expr))"，括
   号()可以用于描述算术优先级。~~ 后续更新：注意，当前版本只能用且（&），不能使用或（|）、非（~）以及标志优先级的括号()。
3. 代数计算。5种基础的代数计算为：加（Add(expr_1,expr_2,...,expr_n)）、减（Sub(expr_1,expr_2)）、乘（Mul(expr_1,expr_2,...,expr_n)）、
   除（Div(expr_1,expr_2)）、幂（Pow(expr_1,expr_2)）。此外，为了简化GDL中代数关系的定义，我们也可以定义一些扩展的代数计算，比如两点之间
   的距离定义为DPP(expr_x1,expr_y1,expr_x2,expr_y2)，来代替Pow(Add(Pow(Sub(expr_y2,expr_y1),2),Pow(Sub(expr_x2,expr_x1),2)),1/2)。
   扩展的代数计算：两点之间的距离DPP(expr_x1,expr_y1,expr_x2,expr_y2)、点到直线的距离DPL(expr_x,expr_y,expr_k,expr_b)、
   两线的夹角MA(expr_k1,expr_k2)、点的幂PP(expr_x,expr_y,expr_a,expr_b,expr_r)
4. 实体参数和实数。只有实体的参数可以用于当前约束的定义，注意区分`constraints`定义的代数约束和`extend`中定义的代数关系的区别。`constraints`
   定义的代数约束只包含实体参数，用于构图过程；`extend`中定义的代数关系只包含实体属性，用于推理过程。

使用角相等的定义说明代数约束和代数关系的区别：  

    "EqualAngle(a,b,l,k)": {
      "type": "basic",
      "ee_check": [
        "Line(a)",
        "Line(b)",
        "Line(l)",
        "Line(k)"
      ],
      "extend": [
        "EqualAngle(l,k,a,b)",
        "Eq(Sub(ab.ma,lk.ma))"
      ],
      "implicit_entity": [],
      "constraints": "Eq(Sub(MA(a.k,b.k),MA(l.k,k.k)))"
    }

上述是两个角相等的关系。在`extend`中，代数关系`Eq(Sub(ab.ma,lk.ma))`中的`ab`表示线a和线b的夹角，`.ma`是属性的符号表示，即角的大小。在
`constraints`中，代数约束`Eq(Sub(MA(a.k,b.k),MA(l.k,k.k)))`中的`MA`是我们自己定义的扩展运算，由两条直线的斜率求夹角，`.k`
是参数的符号表示，这里表示线的斜率。

#### 组合关系（Composite）

组合关系由基础关系通过逻辑运算组合而来。一个组合关系的示例为：

    "MidpointOfArc(M,O,A,B)": {
      "type": "composite",
      "ee_check": [
        "Point(M)",
        "Circle(O)",
        "Point(A)",
        "Point(B)"
      ],
      "extend": [],
      "implicit_entity": [],
      "constraints": "PointOnCircle(M,O)&EqualDistance(M,A,M,B)&PointLeftSegment(M,B,A)"
    }

在`constraints`部分，约束的定义不是直接使用代数约束，而是组合了已有的关系PointOnCircle、EqualDistance和PointLeftSegment。

#### 间接关系（Indirect）

间接关系在组合关系的基础上，使用隐实体间接传导依赖关系。一个间接关系的示例为：

    "IncenterOfTriangle(O,A,B,C)": {
      "type": "indirect",
      "ee_check": [
        "Point(O)",
        "Point(A)",
        "Point(B)",
        "Point(C)"
      ],
      "implicit_entity": [
        "Line(c):PointOnLine(A,c)&PointOnLine(B,c)",
        "Line(a):PointOnLine(B,a)&PointOnLine(C,a)",
        "Line(b):PointOnLine(C,b)&PointOnLine(A,b)",
        "Line(x):EqualAngle(b,x,x,c)",
        "Line(y):EqualAngle(c,y,y,a)",
      ],
      "constraints": "PointLeftSegment(C,A,B)&PointOnLine(O,x)&PointOnLine(O,y)"
    }

在`constraints`部分，构造了隐实体Line(c)、Line(a)、Line(b)、Line(x)、Line(y)来传导点O和点A、B、C之间的依赖关系。

间接关系实际上实现了非线性构造，也就是一个构图语句可以同时求得多个实体。以上述三角形内心的定义为例，实际上是定义了4个点和5条线之间的约束关系。在构
图过程中，该语句可以得到1个未知点和5条线。为了计算尽可能容易，我们要求尽可能线性构图，也就是在应用构图语句时，每条构图语句的参数要求包含且仅包含一
个未知实体。

### 定理（Theorem）

定理是系统的推理规则，分为基础公理和扩展定理两类。

#### 基础公理（Basic）

一个基础公理的定义为：

    "parallel_property_angle_equal(a,l,k)": {
      "type": "basic",
      "ee_check": [
        "Line(a)",
        "Line(l)",
        "Line(k)"
      ],
      "ac_check": "",
      "premise": "Parallel(l,k)",
      "conclusion": [
        "EqualAngle(l,a,k,a)"
      ],
      "proving": {}
    }

`ee_check`用于实体存在性（或者说是实体依赖性）检查。

`ac_check`是代数约束检查，用于检查定理中出现的实体是否符合相对关系约束，定义此约束时需使用关系来描述，使用几何关系、代数约束和逻辑表达式定义。
如三角形ABC要求点C在AB左侧，使用关系PointLeftSegment(C,A,B)描述。

`premise`是定理的前提，使用几何关系、代数关系和逻辑表达式定义。

`conclusion`是定理的结论，是一个关系的列表，一个定理可以有多个结论。

再给出邻补角定理和EqualAngle判定定理的定义：

    "adjacent_complementary_angle(l,k)": {
      "type": "basic",
      "ee_check": [
        "Line(l)",
        "Line(k)"
      ],
      "ac_check": "",
      "premise": "",
      "conclusion": [
        "Eq(Sub(Add(MeasureOfAngle(l,k),MeasureOfAngle(k,l)),180))"
      ],
      "proving": {}
    }

    "equal_angle_judgment(l,k,a,b)": {
      "type": "basic",
      "ee_check": [
        "Line(l)",
        "Line(k)",
        "Line(a)",
        "Line(b)"
      ],
      "ac_check": "",
      "premise": "Eq(Sub(MeasureOfAngle(l,k),MeasureOfAngle(a,b)))",
      "conclusion": [
        "EqualAngle(l,k,a,b)"
      ],
      "proving": {}
    }

#### 扩展定理（Extend）

扩展定理在基础公理的基础上定义，基于以上3个定理的定义，可得：

    "parallel_property_angle_equal_extend(a,l,k)": {
      "type": "extend",
      "ee_check": [
        "Line(a)",
        "Line(l)",
        "Line(k)"
      ],
      "ac_check": "",
      "premise": "Parallel(l,k)",
      "conclusion": [
        "EqualAngle(a,l,a,k)"
      ],
      "proving": {
        "START": [
          "parallel_property_angle_equal(a,l,k)"
        ],
        "parallel_property_angle_equal(a,l,k)": [
          "adjacent_complementary_angle(l,a)",
          "adjacent_complementary_angle(k,a)"
        ],
        "adjacent_complementary_angle(l,a)": [
          "equal_angle_judgment(a,l,a,k)"
        ],
        "adjacent_complementary_angle(k,a)": [
          "equal_angle_judgment(a,l,a,k)"
        ]
      }
    }

`proving`以有向无环图的形式，给出了扩展定理的证明过程。

## 构图过程与推理过程

### 构图语句格式

### construct函数执行过程

1.将间接约束和组合约束，替换为直接约束。替换过程中，对于出现的**隐实体**和**临时实体**，新建其对象，并对临时实体添加相应的约束。替换时，需添加
括号表示运算的优先级。 （其实临时实体就是隐式实体，所有的临时实体都转化为隐实体处理。）

如对于隐实体'Point(X): Relation3(A,B,X)'，新建对象Point(X)，坐标X.x和X.y为未知数 (隐实体可以不用求解，不用添加，因为它是隐式构建的，
所以我们不关心它存不存在，我们只关心最终的实体；但话又说回来，不添加的话会错过很多性质，最好还是添加)
如对于临时实体Line(AB)，如果已经存在实体本身，将其替换为实体本身，否则新建对象Line(l)，并添加约束PointOnLine(A,l) &
PointOnLine(B,l)（临时实体需要添加吗？用的；比如Angle(AB,AC)=60
这个，如果线不存在，那性质也无法添加）

2.将直接约束式转化为析取范式: A&B&C | C&D&E | ... 之后将其解析为符号表示

3.将每一个析取范式转化为符号表示；汇总关系集合；汇总依赖实体集合

4.求解所有约束，得到并保存实体的解，包括目标实体、隐实体和~~临时实体~~。

5.依赖实体为前提，构图语句为边，目标实体、隐实体、临时实体、关系集合为结点，添加条件

### 定理执行语句格式

### apply函数执行过程

1.解析定理（FV check）
2.实体存在性检查 EE check
3.符号替换得前提和结论

operation_id 其实等于 group_id