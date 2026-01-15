"""
根据推理过程的输出，生成新的relation和定理
核心是修改定理库，因为如果定理库不变，仅仅生成新的关系，并没有意义（用于构图的新关系已经生成了，而推理过程并没有使用到生成的新关系）。
考察定理库的性能：
证明的数量：应用定理的数量
复杂度：定理总数量
可推广性：定理库能解决的题目数（无法改变）
次序关系：有一部分定理是自动执行，而不需要预测？
单个定理的前提数量。
第一种定理库的修改，是增加composite关系，替换原先存在该关系的定理的前提，增加结论为该关系的定理
证明的数量↑，复杂度↑
第二种定理库的修改，是合并常用定理组（认为出现辅助构造多的好）
证明的数量↓，复杂度↑
第三种定理库的修改，是合并同前提定理
证明的数量=，复杂度↓
第四种定理库的修改，是增加composite/indirect关系，描述基础构图相关几何实体。替换原先同基础构图的定理的结论。增加前提为该composite/indirect关系的定理。
证明的数量↑，复杂度↑
"""
"""
分析超树的结构：
hypergraph = {
        'notes': [],  # node i, node_id is same with fact_id
        'dependent_entities': [],  # dependent entities of node i
        'edges': [],  # edge i, edge_id is not same with operation_id
        'hypergraph': []  # ((head_node_ids,), edge_id, (tail_node_ids,))
    }
"""

"""
分析能生成定理的超树结构
存在唯一一个note的出度为0
从这个节点回溯到入度为0的节点(×，以hyper1为例，('Point', 'C')的入度不为0，但这个点应该是回溯的终点了)
所有的构图关系,记为集合O
从这个节点通过超边回溯到标签为构图生成的fact的节点(推理出这个fact的必要构图关系，含有辅助线,不含有不等式)，记为集合A。
从这个节点通过dependent_entities回溯到标签为构图生成的fact的节点(构造出这个fact的必要构图关系，不含辅助线,含有不等式)，记为集合B。
由此生成定理：
"D?(集合B中涉及实体)": {
      "type": "new",
      "ee_checks": [
      集合B中涉及实体
      ],
      "ac_checks": 集合B中不等式关系,
      "premises": 集合B中等式关系,
      "conclusions": 出度为0的note,
      "proving": {集合A-集合B,辅助构造}
"""

"""
如果两个定理的集合B完全等价(如何判断？存在同类实体的字母替换映射)，出度为0的note不一样，则这两个定理可以合并为一个定理
"""

"""
分析能生成谓词的超树簇
如果存在一个同类实体的字母替换映射规则，使得定理1的等式前提都在定理2的结论+等式前提中，定理式2的等式前提都是定理1的结论+等式前提中，定理1的不等式与定理2的不等式关系一致(如何衡量？存在同类实体的字母替换映射)
###则认为定理1与定理2的结论可以修改为谓词形式。同时增加定理3，以该谓词形式为前提，以定理1、定理2的结论并集为结论###
"""

"""
什么是一个好的定理？
辅助构造数量a多
定理结论数量b多（需要定理合并）
定理前提数量c少
score=a*b/c
"""

"""
什么是一个好的relation？
什么是relation
等价的relation多
该relation相关的实体少

(推理相关)
什么是relation：一个定理的前提(要求所有的定理均合并完毕,一个定理对应一个relation)
a=等价的relation数量：定理1的等式前提都在定理2的结论+等式前提中；定理2的等式前提都在定理1的结论+等式前提中；定理1的不等式与定理2的不等式关系一致；定理1的等式前提！=定理2的等式前提
b=该relation相关的实体：一个定理的ee_checks
好relation
a多
b少
score=a/b
是否有必要根据relation修改定理（未实现）
(推理无关)（未实现）
什么是relation：一个构图序列，中间辅助构造必须存在
a=等价的relation：一个构图序列与另一个构图序列存在实体解相同，且辅助构造数量不同
b=该relation相关的实体：最终改写的relation变量数目
a多
b少
score=a/b
"""
"""
定理重复的结论要去掉计数（未实现）
修改接口，改成传入题目序号+目标节点信息，使得class Updategdl允许跨题目（录入多个theorem）（未实现）
评分函数修改
测试接口，方便判断入度为0的节点是否一样（3个文件，节点，我的setB，我的setA；待判断的setB，待判断的setA=None，表示不判断该集合）
"""
import ast
from pprint import pprint
from itertools import permutations
import math
import time

def split_rule(rule: str):
    """
    将一条规则字符串按 ':' 和 '&' 拆分为若干谓词元组
    """
    result = []

    # 先按 ':' 拆
    left, right = rule.split(':', 1)

    # 左侧一定是一个谓词，如 Point(D)、Line(s)
    name, args = left.split('(', 1)
    args = args.rstrip(')').split(',')
    result.append((name, *args))

    # 右侧按 '&' 拆成多个谓词
    for part in right.split('&'):
        name, args = part.split('(', 1)
        args = args.rstrip(')').split(',')
        result.append((name, *args))

    return result
def get_setO(hypergraph_data,operations):
    setO=[]
    with_colon = [operation for operation in operations if ':' in operation]
    #print("",with_colon)
    flat_preds = [p for r in with_colon for p in split_rule(r)]
    #print(flat_preds)
    for j in flat_preds:
        #print(j)
        for i in range(0,len(hypergraph_data['notes'])):
            #print(hypergraph_data['notes'][i])
            if hypergraph_data['notes'][i] == j:
                setO.append(i)
                break
    #print("1",set(setO))
    return set(setO)
def get_setO_fast(group,operations):
    #其实没有加速作用，就当作是验证了
    setO = []
    with_colon_index = [i for i, operation in enumerate(operations) if ':' in operation]
    for i in with_colon_index:
        #print(type(ast.literal_eval(group[i])))
        for j in ast.literal_eval(group[i]):
            #print(j)
            setO.append(j)
    #print("11",set(setO))
    return set(setO)
def get_setA(note,fact_notes_id,hypergraph_data):
    for i in range(0,len(hypergraph_data['notes'])):
        if hypergraph_data['notes'][i] == note:
            note_id=i
            break
    setA=[]
    sett=[]
    sett.append(note_id)
    while len(sett)>0:
        x=sett.pop()
        if x in fact_notes_id:
            setA.append(x)
            for edge in hypergraph_data['hypergraph']:
                if x in edge[2]:
                    for j in edge[0]:
                        sett.append(j)
        else:
            for edge in hypergraph_data['hypergraph']:
                if x in edge[2]:
                    for j in edge[0]:
                        sett.append(j)
    return set(setA)
def is_subtuple(sub, tup):
    n, m = len(sub), len(tup)
    return any(tup[i:i+n] == sub for i in range(m - n + 1))
def split_predicates(t):
    res = []
    cur = []

    for x in t:
        if x in (':', '&'):
            if cur:
                res.append(tuple(cur))
                cur = []
        else:
            cur.append(x)

    if cur:
        res.append(tuple(cur))

    return res
def get_setB(note,fact_notes_id,hypergraph_data):
    for i in range(0,len(hypergraph_data['notes'])):
        if hypergraph_data['notes'][i] == note:
            note_id=i
            break
    setB=[]
    settB=[]
    sett=[]
    sett.append(note_id)
    while len(sett)>0:
        #print("set",set)
        x=sett.pop()
        if x in fact_notes_id:
            settB.append(x)
            for j in hypergraph_data['dependent_entities'][x]:
                sett.append(j)
        else:
            for j in hypergraph_data['dependent_entities'][x]:
                sett.append(j)
    #set(settB)表示依赖的实体
    for i in set(settB):
        sub=hypergraph_data['notes'][i]
        for tup in hypergraph_data['edges']:
            if is_subtuple(sub, tup):
                #根据实体找到完整的构图语句tup 如('Line', 's', ':', 'PointOnLine', 'C', 's', '&', 'PointOnLine', 'B', 's')
                for i in range(0, len(hypergraph_data['notes'])):
                    for j in split_predicates(tup):#拆分tup为若干fact
                        if hypergraph_data['notes'][i] == j:
                            setB.append(i)#对应fact的序号加入setB
    #print("setB+note", set(setB),note)
    return set(setB)
def showsetandtheorem(note,hypergraph_data,operations,group):
    """
    展示set
    返回theorem
    """
    theorem={
        "type": "new",
        "ee_checks": [],
        "ac_checks": [],
        "premises": [],
        "conclusions": [note],
        "prove":[]
    }
    """
    print("############集合O")
    start_time = time.time()
    print("集合O长度", len(get_setO()))
    for i in get_setO():
        print(hypergraph_data['notes'][i])
    end_time = time.time()  # 结束计时
    print(f"集合O用时: {end_time - start_time:.10f} 秒")

    start2 = time.time()
    print("集合O'长度", len(get_setO_fast()))
    for i in get_setO_fast():
        print(hypergraph_data['notes'][i])
    end2 = time.time()
    print(f"集合O'用时用时: {end2 - start2:.10f} 秒")
    """
    if get_setO(hypergraph_data,operations)!=get_setO_fast(group,operations):
        raise ValueError("O与O'不一致")
    #start3 = time.time()
    fact_notes_id=get_setO(hypergraph_data,operations)
    #end3 = time.time()
    #print(f"集合O实际用时用时: {end3 - start3:.10f} 秒")
    """
    print("############集合O")
    print("集合O长度",len( get_setO(hypergraph_data,operations) ))
    for i in get_setO(hypergraph_data,operations):
        print(hypergraph_data['notes'][i])
    print("############集合A")
    print("集合A长度",len(get_setA(note,fact_notes_id,hypergraph_data)))
    for i in get_setA(note,fact_notes_id,hypergraph_data):
        print(hypergraph_data['notes'][i])
        
    print("############集合B")
    print("集合B长度",len(get_setB(note,fact_notes_id)))
    for i in get_setB(note,fact_notes_id):
        print(hypergraph_data['notes'][i])
    """
    for i in get_setB(note,fact_notes_id,hypergraph_data):
        #print(hypergraph_data['notes'][i])
        if hypergraph_data['notes'][i][0] in ["Point", "Line", "Circle"]:
            theorem["ee_checks"].append(hypergraph_data['notes'][i])
        elif hypergraph_data['notes'][i][0] in ["PointLeftSegment"]:
            theorem["ac_checks"].append(hypergraph_data['notes'][i])
        elif hypergraph_data['notes'][i][0] in ["FreePoint","FreePoint"]:
            pass
        else:
            theorem["premises"].append(hypergraph_data['notes'][i])
    """
    print("############集合交集")
    for i in get_setB(note,fact_notes_id)&get_setA(note,fact_notes_id):
        print(hypergraph_data['notes'][i])
    print("############集合并集")
    for i in get_setB(note,fact_notes_id) | get_setA(note,fact_notes_id):
        print(hypergraph_data['notes'][i])
    print("############集合A-集合B,辅助构造")
    """
    for i in get_setA(note,fact_notes_id,hypergraph_data)-get_setB(note,fact_notes_id,hypergraph_data):
        #print(hypergraph_data['notes'][i])
        theorem["prove"].append(hypergraph_data['notes'][i])
    """
    print("############集合B-集合A，不等式前提")
    for i in get_setB(note,fact_notes_id)-get_setA(note,fact_notes_id):
        print(hypergraph_data['notes'][i])
    print("$$$$$$$$$$$$$$$$$")
    """
    return theorem
def isequaltheorem(theorem1,theorem2):
    """
    判断两个定理是否可以合并为一个新定理
    如果可以，返回新定理
    如果不行，返回False
    """
    theorem = False
    #生成所有的映射规则
    ee1 = theorem1["ee_checks"]
    ee2 = theorem2["ee_checks"]
    # 按类型分组
    points1 = [n for t, n in ee1 if t == 'Point']
    points2 = [n for t, n in ee2 if t == 'Point']
    lines1 = [n for t, n in ee1 if t == 'Line']
    lines2 = [n for t, n in ee2 if t == 'Line']
    circles1 = [n for t, n in ee1 if t == 'Circle']
    circles2 = [n for t, n in ee2 if t == 'Circle']
    # 基本合法性检查
    if len(points1) != len(points2) or len(lines1) != len(lines2) or len(circles1) != len(circles2):
        mappings = []  # 不可能有映射
    else:
        mappings = []

        # Point 的所有双射
        for p_perm in permutations(points2):
            p_map = dict(zip(points1, p_perm))
            for l_perm in permutations(lines2):
                l_map = dict(zip(lines1, l_perm))
                for c_perm in permutations(circles2):
                    c_map = dict(zip(circles1, c_perm))

                    mapping = {}
                    mapping.update(p_map)
                    mapping.update(l_map)
                    mapping.update(c_map)
                    mappings.append(mapping)
    #print(mappings[0])
    #找到符合要求的映射（theorem1["premises"]通过map与theorem2["premises"]完全一致）
    mapped = set()
    mapped_conclusions = []
    for map in mappings:
        for premise in theorem1["premises"]:
            pred = premise[0]
            args = tuple(map[x] for x in premise[1:])
            mapped.add((pred, *args))
        #print("mapped",mapped)
        if mapped == set(theorem2["premises"]):
            #print("mapped",mapped)
            for concl in theorem1["conclusions"]:
                pred = concl[0]
                args = tuple(map[x] for x in concl[1:])
                mapped_conclusions.append((pred, *args))
            break
        else:
            #print(mapped-set(theorem2["premises"]))
            mapped = set()
    #print("mapped_conclusions",mapped_conclusions)
    #print("theorem2[conclusions]",theorem2["conclusions"])
    if not mapped_conclusions:
        return False
    if mapped_conclusions == theorem2["conclusions"]:
        #print("eq")
        return "EQ"
    theorem = dict(theorem2)
    theorem["conclusions"] = list(set(theorem2["conclusions"]) | set(mapped_conclusions))
    #print("theorem[conclusions]",theorem["conclusions"])
    return theorem
def isnewrelation(theorem1,theorem2):
    """
    判断两个定理是否可以生成一个新关系
    """
    ee1 = theorem1["ee_checks"]
    ee2 = theorem2["ee_checks"]
    # 按类型分组
    points1 = [n for t, n in ee1 if t == 'Point']
    points2 = [n for t, n in ee2 if t == 'Point']
    lines1 = [n for t, n in ee1 if t == 'Line']
    lines2 = [n for t, n in ee2 if t == 'Line']
    circles1 = [n for t, n in ee1 if t == 'Circle']
    circles2 = [n for t, n in ee2 if t == 'Circle']
    # 基本合法性检查
    if len(points1) != len(points2) or len(lines1) != len(lines2) or len(circles1) != len(circles2):
        mappings = []  # 不可能有映射
    else:
        mappings = []

        # Point 的所有双射
        for p_perm in permutations(points2):
            p_map = dict(zip(points1, p_perm))
            for l_perm in permutations(lines2):
                l_map = dict(zip(lines1, l_perm))
                for c_perm in permutations(circles2):
                    c_map = dict(zip(circles1, c_perm))

                    mapping = {}
                    mapping.update(p_map)
                    mapping.update(l_map)
                    mapping.update(c_map)
                    mappings.append(mapping)
    # print(mappings[0])
    # 找到符合要求的映射
    for map in mappings:
        # 要求1：（theorem1["premises"]通过map在theorem2["premises"]+theorem2["conclusions"]里）
        mapped1 = set()
        for premise in theorem1["premises"]:
            pred = premise[0]
            args = tuple(map[x] for x in premise[1:])
            mapped1.add((pred, *args))
        if mapped1 <= set(theorem2["premises"]) | set(theorem2["conclusions"]) and mapped1 !=set(theorem2["premises"]):
            #print("map:",map,"要求1通过")
            # 要求2：（theorem2["premises"]通过反向map在theorem1["premises"]+theorem1["conclusions"]里）
            mapped2 = set()
            inverse_map = {v: k for k, v in map.items()}
            for premise in theorem2["premises"]:
                pred = premise[0]
                args = tuple(inverse_map[x] for x in premise[1:])
                mapped2.add((pred, *args))
            if mapped2 <= set(theorem1["premises"]) | set(theorem1["conclusions"]) and mapped2 !=set(theorem1["premises"]):
                #print("map:",map,"要求2通过")
                #要求3：（theorem1["ac_checks"]通过map与theorem2["ac_checks"]完全一致
                mapped3 = set()
                for premise in theorem1["ac_checks"]:
                    pred = premise[0]
                    args = tuple(map[x] for x in premise[1:])
                    mapped3.add((pred, *args))
                if mapped3 == set(theorem2["ac_checks"]):
                    #print("map:",map,"能生成新谓词")
                    return True
    return False
class Updategdl:
    def __init__(self):
        self.alltheorem={}
        self.allrelation = {}
    def addtheorem(self,theorem_name,theorem):
        if theorem_name in self.alltheorem:
            return False
        self.alltheorem[theorem_name]=theorem
        return True
    def deletetheorem(self,theorem_name):
        if theorem_name in self.alltheorem:
            del self.alltheorem[theorem_name]
            return True
        else:
            return False
    def updatetheorem(self):
        # 要在所有theorem均生成完毕
        changed = True
        while changed:
            changed = False
            keys = list(self.alltheorem.keys())#所有的定理名字
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ki, kj = keys[i], keys[j]#取出来的两个定理名字
                    # 可能在前面的合并中已经被删除
                    if ki not in self.alltheorem or kj not in self.alltheorem:
                        continue

                    c = isequaltheorem(
                        self.alltheorem[ki],
                        self.alltheorem[kj]
                    )

                    if c:
                        if c == "EQ":
                            self.deletetheorem(ki)
                            changed = True
                            break  # 退出 j 循环，重新生成快照
                        else:
                            self.deletetheorem(ki)
                            self.deletetheorem(kj)
                            self.addtheorem(ki, c)

                            changed = True
                            break  # 退出 j 循环，重新生成快照
                if changed:
                    break
    def updatetheoremscore(self):
        # 要在所有theorem均更新完毕
        for i in self.alltheorem:
            a=len(self.alltheorem[i]["prove"])#辅助构造数量
            b=len(self.alltheorem[i]["conclusions"])#定理结论数量
            c=len(self.alltheorem[i]["premises"])#定理前提数量
            self.alltheorem[i]["score"]=(math.log2(a)*b)/(c*c+0.1)

    def addrelation(self,relation_name,relation):
        if relation_name in self.allrelation:
            return False
        self.allrelation[relation_name]=relation
        return True
    def deleterelation(self,relation_name):
        if relation_name in self.allrelation:
            del self.allrelation[relation_name]
            return True
        else:
            return False
    def updaterelation(self):
        #初始化self.allrelation;relation与theorem一一对应
        for i in self.alltheorem:
            a={"constraints":self.alltheorem[i]["premises"],
               "ee_check":self.alltheorem[i]["ee_checks"]}
            self.addrelation(i,a)#根据theorem生成对应的relation,保证两者名字对应
            self.allrelation[i]["score_equivalent"]=1 #等价的relation数目
            self.allrelation[i]["score_relatedobjects"]=len(self.alltheorem[i]["ee_checks"])#相关的实体数目
        #处理掉多余的relation
        changed = True
        while changed:
            changed = False
            keys = list(self.allrelation.keys())  # 所有的relation名字
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ki, kj = keys[i], keys[j]  # 取出来的两个relation名字
                    # 可能在前面的合并中已经被删除
                    if ki not in self.allrelation or kj not in self.allrelation:
                        continue
                    # "score_relatedobjects"不同的两个不可能等价
                    if self.allrelation[ki]["score_relatedobjects"]!=self.allrelation[kj]["score_relatedobjects"]:
                        continue
                    c = isnewrelation(
                        self.alltheorem[ki],
                        self.alltheorem[kj]
                    )

                    if c :
                        score1=self.allrelation[ki]["score_equivalent"]
                        score2 = self.allrelation[kj]["score_equivalent"]
                        self.deleterelation(kj)#删掉第二个
                        self.allrelation[ki]["score_equivalent"]=score1+score2
                        changed = True
                        break  # 退出 j 循环，重新生成快照
                if changed:
                    break
    def updaterelationscore(self):
        #要在所有relation均更新完毕
        for i in self.allrelation:
            a=self.allrelation[i]["score_equivalent"]#等价的relation
            b=self.allrelation[i]["score_relatedobjects"]#relation相关的实体
            self.allrelation[i]["score"]=a/(b+0.1)
def gettheorem(hypergraph_json_path,new_problem_operations_path,problem_groups_json_path,note):
    with open(hypergraph_json_path, "r", encoding="utf-8") as f:
        #"get_hypergraph2.json"
        hypergraph_data = ast.literal_eval(f.read())
    with open(new_problem_operations_path, "r", encoding="utf-8") as f:
        #"newproblemoperations2.json"
        operations = [line.strip().strip('"') for line in f]
    group = []
    with open(problem_groups_json_path, "r", encoding="utf-8") as f:
        #"newproblemgroups2.json"
        for line in f:
            line = line.strip()  # 去掉换行符和空格
            group.append(ast.literal_eval(line))  # 空列表 "[]" 也会被解析为 []
    theorem = showsetandtheorem(note,hypergraph_data,operations,group)
    return theorem
    #note = ('AngleEqualAngle', 'a', 'b', 'p', 'q')
def checkset(hypergraph_json_path,new_problem_operations_path,problem_groups_json_path,note,nsetA=False,nsetB=False,nsetO=False):
    with open(hypergraph_json_path, "r", encoding="utf-8") as f:
        #"get_hypergraph2.json"
        hypergraph_data = ast.literal_eval(f.read())
    with open(new_problem_operations_path, "r", encoding="utf-8") as f:
        #"newproblemoperations2.json"
        operations = [line.strip().strip('"') for line in f]
    group = []
    with open(problem_groups_json_path, "r", encoding="utf-8") as f:
        #"newproblemgroups2.json"
        for line in f:
            line = line.strip()  # 去掉换行符和空格
            group.append(ast.literal_eval(line))  # 空列表 "[]" 也会被解析为 []
    if get_setO(hypergraph_data,operations)!=get_setO_fast(group,operations):
        raise ValueError("O与O'不一致")
    fact_notes_id=get_setO(hypergraph_data,operations)
    #print("setO",get_setO(hypergraph_data,operations))
    #print("setA",get_setA(note,fact_notes_id,hypergraph_data))
    #print("setB",get_setB(note, fact_notes_id, hypergraph_data))
    if nsetO is False:
        pass
    else:
        setO=get_setO(hypergraph_data,operations)
        if setO!=nsetO:
            print("nsetO",nsetO)
            print("setO", setO)
            raise ValueError("nsetO与setO不一样")

    if nsetA is False:
        pass
    else:
        setA=get_setA(note,fact_notes_id,hypergraph_data)
        if setA!=nsetA:
            print("nsetA",nsetA)
            print("setA", setA)
            raise ValueError("nsetA与setA不一样")

    if nsetB is False:
        pass
    else:
        setB = get_setB(note, fact_notes_id, hypergraph_data)
        if setB != nsetB:
            print("nsetB", nsetB)
            print("setB", setB)
            raise ValueError("nsetB与setB不一样")

    return "通过检测"


if __name__ == '__main__':
    #msetB={0, 1, 2, 3, 4, 5, 8, 9, 11, 13, 14, 16, 23, 24, 25, 26, 35, 36, 45, 54, 57, 58, 60, 67, 68, 70}
    print(checkset("data/get_hypergraph_num.json", "data/newproblemoperations_num.json", "data/newproblemgroups_num.json", ('SegmentEqualSegment', 'B', 'E', 'D', 'C'), nsetA=[],
                   nsetB=False, nsetO=False))

    """
    使用方式说明：根据要求修改对应文件链接
    需要检测某个集合时，修改False为集合
    note：tuple格式，为需要检测的节点内容，如 ('AngleEqualAngle', 'a', 'b', 'p', 'q')
    nsetO：构造该题目的fact序号集合
    nsetA：构造该题目的fact序号集合中，与推理出note相关的序号集合
    nsetB：构造该题目的fact序号集合中，与构造出note相关的序号集合
    
    "get_hypergraph2.json"构建代码：
    
    get_h = pformat(get_hypergraph(problem,False), width=400)
    file_path4 = f"get_hypergraph{num}.json"
    with open(file_path4, "w", encoding="utf-8") as f:
        f.write(get_h)
    
    "newproblemoperations2.json"构建代码：
    
    file_path2 = f"newproblemoperations{num}.json"
    with open(file_path2, "w", encoding="utf-8") as f:
        for item in problem.operations:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path2}")
    
    "newproblemgroups2.json"构建代码：
    
    file_path5 = f"newproblemgroups{num}.json"
    with open(file_path5, "w", encoding="utf-8") as f:
        for item in problem.groups:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path5}")
    
    """

    theorem1=gettheorem("get_hypergraph2.json","newproblemoperations2.json","newproblemgroups2.json",('AngleEqualAngle', 'a', 'b', 'p', 'q'))
    theorem2=gettheorem("get_hypergraph2.json","newproblemoperations2.json","newproblemgroups2.json",('AngleEqualAngle', 'a', 'c', 'p', 'r'))
    theorem3 = {
        'ac_checks': [('PointLeftSegment', 'C', 'A', 'B'),
                      ('PointLeftSegment', 'R', 'P', 'Q')],
        'conclusions': [('AngleEqualAngle', 'a', 'c', 'p', 'r')],
        'ee_checks': [('Point', 'A'),
                      ('Point', 'B'),
                      ('Point', 'C'),
                      ('Line', 'a'),
                      ('Line', 'c'),
                      ('Point', 'P'),
                      ('Point', 'Q'),
                      ('Point', 'R'),
                      ('Line', 'p'),
                      ('Line', 'r'),
                      ('Line', 'b'),
                      ('Line', 'q')],
        'premises': [('PointOnLine', 'B', 'a'),
                     ('PointOnLine', 'C', 'a'),
                     ('PointOnLine', 'A', 'c'),
                     ('PointOnLine', 'B', 'c'),
                     ('SegmentEqualSegment', 'A', 'B', 'P', 'Q'),
                     ('SegmentEqualSegment', 'A', 'C', 'P', 'R'),
                     ('SegmentEqualSegment', 'B', 'C', 'Q', 'R'),
                     ('PointOnLine', 'Q', 'p'),
                     ('PointOnLine', 'R', 'p'),
                     ('PointOnLine', 'P', 'r'),
                     ('PointOnLine', 'Q', 'r'),
                     ('PointOnLine', 'A', 'b'),
                     ('PointOnLine', 'C', 'b'),
                     ('PointOnLine', 'P', 'q'),
                     ('PointOnLine', 'R', 'q')
                     ],
        'proving': [('Line', 'l'),
                    ('PointOnLine', 'Q', 'l'),
                    ('AngleEqualAngle', 'c', 'a', 'p', 'l'),
                    ('Line', 'm'),
                    ('PointOnLine', 'R', 'm'),
                    ('AngleEqualAngle', 'm', 'p', 'a', 'b'),
                    ('Point', 'S'),
                    ('PointOnLine', 'S', 'l'),
                    ('PointOnLine', 'S', 'm'),
                    ('Line', 'n'),
                    ('PointOnLine', 'P', 'n'),
                    ('PointOnLine', 'S', 'n'),
                    ('PointOnLine', 'D', 'n'),
                    ('PointOnLine', 'D', 'p')],
        'type': 'new'}
    theorem4 = {
        'ac_checks': [('PointLeftSegment', 'C', 'A', 'B'),
                      ('PointLeftSegment', 'R', 'P', 'Q')],
        'conclusions': [('AngleEqualAngle', 'a', 'b', 'p', 'q')],
        'ee_checks': [('Point', 'A'),
                      ('Point', 'B'),
                      ('Point', 'C'),
                      ('Line', 'a'),
                      ('Line', 'b'),
                      ('Point', 'P'),
                      ('Point', 'Q'),
                      ('Point', 'R'),
                      ('Line', 'p'),
                      ('Line', 'q'),
                      ('Line', 'c'),
                      ('Line', 'r')],
        'premises': [('PointOnLine', 'B', 'a'),
                     ('PointOnLine', 'C', 'a'),
                     ('PointOnLine', 'A', 'b'),
                     ('PointOnLine', 'C', 'b'),
                     ('SegmentEqualSegment', 'A', 'B', 'P', 'Q'),
                     ('SegmentEqualSegment', 'A', 'C', 'P', 'R'),
                     ('SegmentEqualSegment', 'B', 'C', 'Q', 'R'),
                     ('PointOnLine', 'Q', 'p'),
                     ('PointOnLine', 'R', 'p'),
                     ('PointOnLine', 'P', 'q'),
                     ('PointOnLine', 'R', 'q'),
                     ('PointOnLine', 'A', 'c'),
                     ('PointOnLine', 'P', 'r'),
                     ('PointOnLine', 'Q', 'r'),
                     ('PointOnLine', 'B', 'c')],
        'prove': [('Line', 'l'),
                  ('PointOnLine', 'Q', 'l'),
                  ('AngleEqualAngle', 'c', 'a', 'p', 'l'),
                  ('Line', 'm'),
                  ('PointOnLine', 'R', 'm'),
                  ('AngleEqualAngle', 'm', 'p', 'a', 'b'),
                  ('Point', 'S'),
                  ('PointOnLine', 'S', 'l'),
                  ('PointOnLine', 'S', 'm'),
                  ('Line', 'n'),
                  ('PointOnLine', 'P', 'n'),
                  ('PointOnLine', 'S', 'n'),
                  ('PointOnLine', 'D', 'n'),
                  ('PointOnLine', 'D', 'p')],
        'type': 'new'}
    theorem5 = {'ac_checks': [('PointLeftSegment', 'C', 'A', 'B')],
                'conclusions': [('AngleEqualAngle', 'a', 'b', 'b', 'c')],
                'ee_checks': [('Point', 'A'),
                              ('Point', 'B'),
                              ('Point', 'C'),
                              ('Line', 'a'),
                              ('Line', 'b'),
                              ('Line', 'c')],
                'premises': [('PointOnLine', 'B', 'a'),
                             ('PointOnLine', 'C', 'a'),
                             ('PointOnLine', 'A', 'b'),
                             ('PointOnLine', 'C', 'b'),
                             ('SegmentEqualSegment', 'A', 'B', 'A', 'C'),
                             ('PointOnLine', 'A', 'c'),
                             ('PointOnLine', 'B', 'c')],
                'prove': [('PointOnLine', 'S', 'l'),
                          ('PointOnLine', 'S', 'm')],
                'type': 'new'}
    theorem6 = {'ac_checks': [('PointLeftSegment', 'C', 'A', 'B')],
                'conclusions': [('SegmentEqualSegment', 'A', 'B', 'A', 'C')],
                'ee_checks': [('Point', 'A'),
                              ('Point', 'B'),
                              ('Point', 'C'),
                              ('Line', 'a'),
                              ('Line', 'b'),
                              ('Line', 'c')],
                'premises': [('PointOnLine', 'B', 'a'),
                             ('PointOnLine', 'C', 'a'),
                             ('PointOnLine', 'A', 'b'),
                             ('PointOnLine', 'C', 'b'),
                             ('AngleEqualAngle', 'a', 'b', 'b', 'c'),
                             ('PointOnLine', 'A', 'c'),
                             ('PointOnLine', 'B', 'c')],
                'prove': [('PointOnLine', 'S', 'l'),
                          ('PointOnLine', 'S', 'm')],
                'type': 'new'}
    theorem6_mapped = {
        'ac_checks': [('PointLeftSegment', 'E', 'D', 'C')],
        'conclusions': [('SegmentEqualSegment', 'D', 'C', 'D', 'E')],
        'ee_checks': [('Point', 'D'),
                      ('Point', 'C'),
                      ('Point', 'E'),
                      ('Line', 'a'),
                      ('Line', 'b'),
                      ('Line', 'c')],
        'premises': [('PointOnLine', 'C', 'a'),
                     ('PointOnLine', 'E', 'a'),
                     ('PointOnLine', 'D', 'b'),
                     ('PointOnLine', 'E', 'b'),
                     ('AngleEqualAngle', 'a', 'b', 'b', 'c'),  # Line 不替换
                     ('PointOnLine', 'D', 'c'),
                     ('PointOnLine', 'C', 'c')],
        'prove': [('PointOnLine', 'S', 'l'),
                  ('PointOnLine', 'S', 'm')],
        'type': 'new'
    }
    theorem8 = {'ac_checks': [('PointLeftSegment', 'C', 'A', 'B')],
                'conclusions': [('AngleEqualAngle', 'a', 'c', 'b', 'c')],
                'ee_checks': [('Point', 'A'),
                              ('Point', 'B'),
                              ('Point', 'C'),
                              ('Line', 'a'),
                              ('Line', 'b'),
                              ('Line', 'c')],
                'premises': [('PointOnLine', 'B', 'a'),
                             ('PointOnLine', 'C', 'a'),
                             ('PointOnLine', 'A', 'b'),
                             ('PointOnLine', 'C', 'b'),
                             ('SegmentEqualSegment', 'A', 'B', 'A', 'C'),
                             ('PointOnLine', 'A', 'c'),
                             ('PointOnLine', 'B', 'c')],
                'prove': [('PointOnLine', 'S', 'l'),
                          ('PointOnLine', 'S', 'm')],
                'type': 'new'}
    updategdl = Updategdl()
    updategdl.addtheorem("1", theorem1)
    updategdl.addtheorem("2", theorem2)
    updategdl.addtheorem("3", theorem3)
    updategdl.addtheorem("4", theorem4)
    updategdl.addtheorem("5", theorem5)
    updategdl.addtheorem("6", theorem6)
    updategdl.addtheorem("7", theorem6_mapped)
    updategdl.addtheorem("8", theorem8)
    updategdl.updatetheorem()
    updategdl.updatetheoremscore()
    updategdl.updaterelation()
    updategdl.updaterelationscore()
    print("%%%%%%alltheorem%%%%%")
    pprint(updategdl.alltheorem)
    print("%%%%%%allrelation%%%%")
    pprint(updategdl.allrelation)

