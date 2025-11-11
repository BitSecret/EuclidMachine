"Entity"
点
线
圆
"Measure"
参数(Parameter)(构图系统能访问，推理系统不能访问)
点的横坐标x
点的纵坐标y
线的k
线的b
圆心横坐标u
圆心纵坐标v
圆半径r
属性(Attribution)(构图系统不能访问，推理系统能访问)
两点距离dpp
点线距离dpl
点圆距离dpc
两线夹角ma (lm.ma,l顺时针转到m)
圆半径rad

关系(Relation)
(构图系统能访问"ee_check"，用于判断构图语句是否合法；构图系统能访问"constraints"与"implicit_entity"，用于形成点)
(推理系统能访问extend，用于获取更多的事实)
(基础几何关系的constraints是参数关系)
(只有基础几何关系的extend可以有属性)
(组合关系的constraints是基础几何关系的逻辑运算)
(间接关系的constraints，使用间接关系时，必须把implicit_entity和constraints全部添加进入，这样才能转换为基础几何关系的集合)
(对于发现的新间接关系：若干个不同前提的定理，涉及初始实体相同，推理后初始实体相关的事实集相同，则取其中某一个前提->任意结论，用其涉及的实体，定义出该新间接关系的约束。之后修改相关定理)
列举几何关系：
//
注释：\为已完成的,b表示基础几何关系basic，c表示组合关系composite，i表示间接关系indirect
注释：IP:可作为定理前提 NP:不可作为定理前提
     IC:可作为定理结论 NC:不可作为定理结论
注释：
基础几何关系basic分为：
有前缀--等式关系
无前缀--不等式关系，推理中隐式表现在ac——check


//
点和点：点A,B重合b\
点和线：点A在线l上b\,构成ABC顺时针三角形c\，构成ABC逆时针三角形c\(实际上用不到,用到的所有三角形都是顺时针三角形)
点和线段：点A在线段BC前方i\，点A在线段BC左侧b\，点点A在线段BC右侧b\，点A在线段BC内i\，点A为线段BC中点i\
点和角：点A在角lm内部i\，点A在角lm外部i\
点和圆：点为圆心b\，点在圆上b\，点在圆内b\，点在圆外b\
点和弧：点在弧上c\，点为弧中点c\

点和三角形：点在三角形内c\，点在三角形外c(不用，涉及或)\，外心 三边垂直平分线i\ ，内心 三个内角的角平分线i\，重心 三条中线i\ ，垂心 三条高线i \，ABC旁心 一个内角A的角平分线，另外两个外角的角平分线i\

线和线：线重合b\，线平行b\，线垂直b\
线和线段：线段垂直平分线i\
线和角：角平分线c\
线和圆：相切b\，相离b\，相交b\
线和弧：
线和三角形：

线段和线段：等b\，大于b\
线段和角：
线段和圆：等长半径b\，大于半径b\,小于半径b\
线段和弧：
线段和三角形：

角和角：等b\，大于b\
角和圆：
角和弧：
角和三角形：

圆和圆：圆重叠b\，圆全等b\，外离b\，外切b\，相交b\，内切b\，内含b\
圆和弧：
圆和三角形：外接圆i\，内切圆i\，旁切圆i\
弧和弧：等i\，大于i\
弧和三角形：

三角形和三角形：全等c\，镜像全等c\，相似c\，镜像相似c\

定理：
前提，类别为b的关系
结论，类别为b的关系

分析DD中的定理
见DD文档，
红色表示不需要这个定理，关系的定义中就包含了
黄色表示需要这个定理，但不能直接用
绿色表示需要这个定理
蓝色表示不需要这个定理，纯代数关系的变换
粉色表示暂时不考虑，后续修改(已完成)

分析定理:逻辑推理与代数推理(属性相关的代数运算)的桥梁
     属性(Attribution)(构图系统不能访问，推理系统能访问)
     两点距离dpp
     点线距离dpl
     点圆距离dpc
     两线夹角ma (lm.ma,l顺时针转到m)
     圆半径rad
关于属性的代数运算:
     AB.dpp=BA.DPP
     lm.ma+ml.ma=180
     lm.ma+mn.ma=ln.ma
(NP,IC)SamePoint(A,B)-->AB.dpp=0 (extend)           ;AB.dpp-->SamePoint(A,B) (A1)
(IP,IC)PointOnLine(A,l)-->Al.dpl=0 (extend)         ;Al.dpl-->PointOnLine(A,l) (A2); 
(IP,IC)PointIsCircleCenter(A,O)-->AO.dpc=0 (extend)         ;AO.dpc=0-->PointIsCircleCenter(A,O) (A3)
(IP,IC)PointOnCircle(A,O)-->AO.dpc=O.rad (extend)           ;AO.dpc=O.rad-->PointOnCircle(A,O) (A4)
(NP,IC)SameLine(a,b)-->ab.ma=0 (extend)                     ;无法从代数推理得到
(IP,IC)LinesParallel(a,b)-->ab.ma=0 (extend)                ;ab.ma=0-->LinesParallel(a,b) (A5)
(IP,IC)LinesPerpendicular(a,b)-->ab.ma=90 (extend)          ;ab.ma=90-->LinesPerpendicular(a,b) (A6)
(IP,NC)LineTangentToCircle(l,O)-->Pl.dpl=O.rad LinesPerpendicular(l,m) (DcheckLineTangentToCircle)            ;无法从代数推理得到
(IP,IC)SegmentEqualSegment(A,B,C,D)-->AB.dpp=CD.dpp (extend)               ;AB.dpp=CD.dpp-->SegmentEqualSegment(A,B,C,D) (A7)
(NP,IC)SegmentEqualCircleRadius(A,B,O)-->O.rad=AB.dpp (extend)             ;O.rad=AB.dpp-->SegmentEqualCircleRadius(A,B,O) (A9)
(IP,IC)AngleEqualAngle(a,b,c,d)-->ab.ma=cd.ma (extend)                     ;ab.ma=cd.ma-->AngleEqualAngle(a,b,c,d) (A8)
(NP,IC)SameCircle(O,Q)-->O.rad=Q.rad (extend)               ;无法从代数推理得到
(NP,IC)CirclesCongruent(O,Q)-->O.rad=Q.rad (extend)         ;O.rad=Q.rad-->CirclesCongruent(O,Q) (DcheckCirclesCongruent)
(IP,NC)CirclesExternallyTangent(O,Q)--> AB.dpp=O.rad+Q.rad (DcheckCirclesExternallyTangent(O,Q,A,B))               ;无法从代数推理得到
(IP,NC)CirclesInternallyTangent(O,Q)--> AB.dpp=abs|O.rad-Q.rad| (DcheckCirclesInternallyTangent(O,Q,A,B))          ;无法从代数推理得到
代数推理-->逻辑推理：
(A10,A11,A12)用dpp等式(AB.dpp+BC.dpp=AC.dpp)证明点在线上 

逻辑推理-->代数推理： 
(A13,A14,A15)用点在线上证明dpp等式(AB.dpp+BC.dpp=AC.dpp)
A16用角的构图<PA,PB,PC顺时针>证明ma等式(lm.ma+mn.ma=ln.ma) 
A17用角的构图证明ma等式(ab.ma+ba.ma=180)

代数推理内部桥梁：
dpp<-->dpp (A18)
dpc<-->dpp (DcheckdppAnddpc)
dpl<-->dpp (DcheckdppAnddpl)