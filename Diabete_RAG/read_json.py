import os
import json
import networkx as nx
from pyvis.network import Network

# 1. 【新增】从文件加载数据，解决 all_triplets 未定义的问题
input_file = 'diabetes_kg_linked.json'
if not os.path.exists(input_file):
    raise FileNotFoundError(f"未找到文件 {input_file}，请先运行上半部分代码生成三元组文件。")

with open(input_file, 'r', encoding='utf-8') as f:
    all_triplets = json.load(f)

print(f"成功加载 {len(all_triplets)} 个三元组，开始构建图谱...")

# 2. 初始化图
G = nx.DiGraph()

# 3. 添加节点和边
for t in all_triplets:
    # 防止实体名称中包含下划线导致 ID 混淆，建议使用更安全的分隔符或直接哈希，但此处保持原逻辑
    head_node_id = f"{t['head_type']}_{t['head_entity']}"
    tail_node_id = f"{t['tail_type']}_{t['tail_entity']}"

    G.add_node(head_node_id, label=t['head_entity'], group=t['head_type'], title=f"类型:{t['head_type']}")
    G.add_node(tail_node_id, label=t['tail_entity'], group=t['tail_type'], title=f"类型:{t['tail_type']}")

    # 添加边
    G.add_edge(head_node_id, tail_node_id, label=t['relation'])

# 4. 使用 pyvis 渲染为交互式 HTML
try:
    # 初始化网络
    net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)

    # 获取当前脚本所在绝对路径，确保相对路径正确
    # 有时 pyvis 在当前工作目录找不到模板，显式保存可能有助于规避
    output_filename = "diabetes_knowledge_graph.html"

    print(f"正在生成图谱：{output_filename} ...")
    net.show(output_filename, notebook=False)

    # 验证生成结果
    if os.path.exists(output_filename):
        size = os.path.getsize(output_filename)
        if size > 0:
            print(f"✅ 成功生成图谱 ({size} bytes)：{os.path.abspath(output_filename)}")
        else:
            print("❌ 生成的文件为空，可能是模板渲染失败。")
    else:
        print("❌ 文件未生成。")

except AttributeError as e:
    if "'NoneType' object has no attribute 'render'" in str(e):
        print("❌ 检测到模板渲染错误 (AttributeError)。")
        print("💡 解决方案：请运行 'pip uninstall pyvis jinja2 && pip install --no-cache-dir pyvis jinja2' 修复环境。")
        raise e
    else:
        raise e
except Exception as e:
    print(f"❌ 发生其他错误：{e}")
    raise e
