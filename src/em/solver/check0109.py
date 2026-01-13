import json
import ast
import re
import copy
from sympy import symbols, Eq, sympify
from sympy import sympify, symbols, atan, pi, log
from sympy import parse_expr
from pprint import pprint
from pprint import PrettyPrinter
def extract_entity_names(s):
    """
    输入: "('PointOnLine', ('A', 'B'))"
    输出: ['A', 'B']
    """
    # 匹配第二项括号内容
    match = re.search(r"\(\s*([^()]+)\s*\)", s.split(",", 1)[1])
    if match:
        content = match.group(1)
        # 去掉空格和单引号，然后按逗号分割
        names = [x.strip(" '") for x in content.split(",")]
        # 去掉空字符串
        names = [n for n in names if n]
        return names
    return []
def check_have_wrong_fact(fact_name,entities_name,gdl_name):

    with open(gdl_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 取 "Relations" 项
    gdl = data.get("Relations")

    # 查看结果
    #print("gdl",gdl)

    lines = []
    with open(fact_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 这里每行都是一个完整字符串，不做解析
            line = line.strip('"')
            #print("line",line)
            match = re.match(r"(\('.*?',\s*\(.*?\))", line)
            #print("match",match)
            if match:
                s2 = match.group(1)
                lines.append(s2)
            else:
                #print("match没有")
                lines.append("Equation")

    #print(lines)

    types = set()

    for l in lines:
        # 匹配 '...' 中的第一项
        match = re.match(r"\('([^']+)'", l)
        if match:
            types.add(match.group(1))

    #print(types)
    all_constraint=[]
    for l in lines:
        if l == "Equation":
            all_constraint.append("Equation")
        else:
            type_name = l.split(",")[0].strip("(' ")
            entity_name = extract_entity_names(l)
            #print(type_name,entity_name)
            template_key = None
            #print("%%%type_name",type_name)
            if type_name in ["Line", "Point","Circle"]:
                all_constraint.append("Base")
            else:
                for key, value in gdl.items():
                    #print(key)
                    if key.startswith(type_name + "("):
                        template_key = key
                        # 提取模板里的变量名顺序，从 ee_checks 里取
                        template_vars = []
                        # 遍历所有 ee_checks
                        for ee in value["ee_checks"]:
                            match = re.search(r"\(\s*([^\)]+)\s*\)", ee)
                            if match:
                                vars_inside = match.group(1)  # 括号内的内容
                                # 按逗号拆分，并去掉空格
                                vars_list = [v.strip() for v in vars_inside.split(',') if v.strip()]
                                template_vars.extend(vars_list)

                if template_key is None:
                    raise ValueError(f"No template found for type {type_name}")
                template_constraints = gdl[template_key]['constraints']

                if template_constraints:
                    new_constraints = copy.deepcopy(template_constraints)
                    placeholder = {}
                    #  先用唯一占位符替换模板变量
                    # 注意：只匹配完整变量名，避免误伤其他字符
                    for i, var in enumerate(template_vars):
                        ph = f"__VAR{i}__"
                        placeholder[var] = ph
                        # 用正则替换，确保只替换完整变量，不影响外部字母
                        new_constraints = re.sub(r'\b{}\b'.format(re.escape(var)), ph, new_constraints)

                    # 再把占位符替换成实际实体名
                    for var, name in zip(template_vars, entity_name):
                        new_constraints = new_constraints.replace(placeholder[var], name)
                    all_constraint.append(new_constraints)
                else:
                    all_constraint.append("Empty")

    #print("all_constraint",all_constraint)
    with open(entities_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    # 处理 Points
    for pt_name, coords_list in data.get("Point", {}).items():
        coords = coords_list[0]  # 取第一项
        mapping[f"{pt_name}.x"] = coords[0]
        mapping[f"{pt_name}.y"] = coords[1]

    # 处理 Lines
    for line_name, params_list in data.get("Line", {}).items():
        params = params_list[0]  # 取第一项
        mapping[f"{line_name}.k"] = params[0]
        mapping[f"{line_name}.b"] = params[1]
    # 处理 Circles
    for circle_name, params_list in data.get("Circle", {}).items():
        params = params_list[0]  # 取第一组参数 [cx, cy, r]
        mapping[f"{circle_name}.cx"] = params[0]
        mapping[f"{circle_name}.cy"] = params[1]
        mapping[f"{circle_name}.r"] = params[2]

    #print(mapping)

    def calc_operation(operation, paras):
        if operation == 'Add':
            result = paras[0]
            for p in paras[1:]:
                result += p
        elif operation == 'Sub':
            result = paras[0] - paras[1]
        elif operation == 'Mul':
            result = paras[0]
            for p in paras[1:]:
                result *= p
        elif operation == 'Div':
            result = paras[0] / paras[1]
        elif operation == 'Pow':
            result = paras[0] ** paras[1]
        elif operation == 'ABS':
            result = abs(paras[0])
        elif operation == 'DPP':  # DPP(x1,y1,x2,y2)
            result = (paras[2] - paras[0]) ** 2 + (paras[3] - paras[1]) ** 2
        elif operation == 'DPL':  # DPL(x,y,k,b)
            result = (paras[2] * paras[0] - paras[1] + paras[3]) ** 2 / (paras[2] ** 2 + 1)
        elif operation == 'MA':  # MA(k1,k2)
            result = (atan(paras[0]) - atan(paras[1])) * 180 / pi
        elif operation == 'MAM':  # MAM(k1,k2)
            result = ((atan(paras[0]) - atan(paras[1])) * 180 / pi + 180) % 180
        elif operation == 'PP':  # PP(x,y,cx,cy,r)
            result = (paras[2] - paras[0]) ** 2 + (paras[3] - paras[1]) ** 2 - paras[4] ** 2
        elif operation == 'Log':  # Log(x)
            result = log(paras[0])
        else:
            raise Exception(f"Unknown operation '{operation}'")
        return result

    # 递归解析表达式
    def parse_expr(expr_str):
        # 如果是数字
        try:
            return float(expr_str)
        except:
            pass
        # 如果是映射里的变量
        if expr_str in mapping:
            return mapping[expr_str]

        # 匹配函数格式 Operation(p1,p2,...)
        import re
        match = re.match(r'(\w+)\((.*)\)', expr_str)
        if match:
            op = match.group(1)
            paras_str = match.group(2)
            # 拆分参数，支持嵌套
            paras = []
            depth = 0
            current = ''
            for c in paras_str:
                if c == ',' and depth == 0:
                    if current.strip():
                        paras.append(parse_expr(current.strip()))
                    current = ''
                else:
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                    current += c
            if current.strip():
                paras.append(parse_expr(current.strip()))
            return calc_operation(op, paras)

    # 遍历 all_constraint
    results = []
    EPS = 1e-2
    for idx, constr in enumerate(all_constraint):
        #print("constr", constr)
        if constr.startswith(("Eq(", "L(","G("))  and constr.endswith(")"):
            parts = constr.split("&")
            sub_results = []
            for part in parts:
                part = part.strip()
                if part.startswith("Eq(") and part.endswith(")"):
                    inner = part[3:-1]
                    val = float(parse_expr(inner))
                    if lines[idx].split(",")[0].strip("()'\" ") in ["AngleEqualAngle"]:
                        #print("^^^^^^",type(val))
                        #print("######",abs(val)%180.0)
                        #print("######", (abs(val) % 180.0)< EPS)
                        a=(abs(val)%180.0 < EPS) or (abs(val)%180.0-180.0 < EPS)
                        sub_results.append(a)
                    else:
                        sub_results.append(abs(val) < EPS)


                elif part.startswith("G(") and part.endswith(")"):
                    # 需要修改!!!!!!!!!!!
                    inner = part[2:-1]
                    val = float(parse_expr(inner))
                    sub_results.append(val > EPS)

                elif part.startswith("L(") and part.endswith(")"):
                    # 需要修改!!!!!!!!!!!
                    inner = part[2:-1]
                    val = float(parse_expr(inner))
                    sub_results.append(val < -EPS)
            results.append(all(sub_results))
        elif constr in ["Base","Empty","Equation"]:
            results.append("ok")
        elif lines[idx].split(",")[0].strip("()'\" ") in ["ClockwiseTriangle"]:
            results.append("ClockwiseTriangle")
        else:
            raise ValueError("解析失败")
    pp = PrettyPrinter(width=100, compact=True)
    pp.pprint(results)

    print("len(results)",len(results))
    print("len(lines)", len(lines))
    for i in range(0,len(results)):
        if results[i] is False:
            print("有错误fact")
            print(lines[i])


def check_two_fact_json(fact_name1, fact_name2):
    """
    比较两个 fact 文件，每行是字符串：
    "('Point', ('A',), (), (), 0)"
    忽略倒数第二个 '(' 后面的内容，只比较前面部分。
    """
    def preprocess(lines):
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 找到倒数第二个 '(' 的索引
            idxs = [i for i, c in enumerate(line) if c == '(']
            if len(idxs) >= 2:
                cutoff = idxs[-2]
                new_line = line[:cutoff]
            else:
                new_line = line  # 括号少于 2 个，整行保留
            new_lines.append(new_line)
        return set(new_lines)

    # 读取文件
    with open(fact_name1, "r", encoding="utf-8") as f1:
        lines1 = [line.strip() for line in f1 if line.strip()]
    with open(fact_name2, "r", encoding="utf-8") as f2:
        lines2 = [line.strip() for line in f2 if line.strip()]

    # 处理忽略倒数第二个 '(' 后面的内容
    set1 = preprocess(lines1)
    set2 = preprocess(lines2)

    # 找出差异
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    if not only_in_1 and not only_in_2:
        return True
    else:
        return {"only_in_1": only_in_1, "only_in_2": only_in_2}

if __name__ == '__main__':
    # check_have_wrong_fact(fact_name="newproblemfacts2_szh.json", entities_name="newproblementities2.json", gdl_name="gdl-yuchang.json")
    # pprint(check_two_fact_json(fact_name1="newproblemfacts2.json", fact_name2="newproblemfacts2_szh.json"))
    check_have_wrong_fact(fact_name="E:\PythonWorkSpace\EuclidMachine\src\em\solver\\newproblemfacts2.json", entities_name="E:\PythonWorkSpace\EuclidMachine\src\em\solver\\newproblementities2.json", gdl_name="E:\PythonWorkSpace\EuclidMachine\data\\new_gdl\gdl-yuchang.json")
    pprint(check_two_fact_json(fact_name1="E:\\newproblemfacts2.json", fact_name2="E:\PythonWorkSpace\EuclidMachine\src\em\solver\\newproblemfacts2.json"))