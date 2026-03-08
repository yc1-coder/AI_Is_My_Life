"""
一、本代码用于从 JSON 文件中提取三元组数据，并保存为新的 JSON 文件。

核心逻辑
遍历文件：保持原有的文件读取逻辑。
定位数据：进入 data['paragraphs'] -> item['sentences']。
构建映射：为了通过 entity_id 快速找到实体名称和类型，先在句子级别建立一个 id 到 entity_info 的映射字典。
生成三元组：遍历 relations 列表，通过 head_entity_id 和 tail_entity_id 从映射中获取具体的实体信息，组装成三元组。

代码关键点解析
entity_map 字典：这是最关键的一步。原始数据中关系只存储了 T1374 这样的 ID，我们需要先遍历 entities 列表，把 ID 和具体的词（如 "CKD"）对应起来，才能在处理 relations 时还原出完整的三元组。
容错处理：增加了 KeyError 捕获，防止因某个 JSON 文件结构略有不同（例如缺少 sentences 字段）导致程序崩溃。
数据丰富度：生成的三元组不仅包含 (头，关系，尾)，还保留了 head_type（实体类型）、source_file（来源文件）等信息，便于后续分析或构建知识图谱。
输出格式：最终结果保存为 diabetes_kg_triplets.json，是一个扁平化的列表，非常适合直接导入 Neo4j 或其他图数据库。
你可以直接运行这段代码，它将自动扫描 diabetes_data 文件夹下的所有 JSON，并输出转换后的三元组文件。
"""
import os
import json
import pprint
from natsort import natsorted

folder_path = 'diabetes_data'
all_triplets = []


def extract_triplets_from_paragraph(paragraph_data, filename):
    """
    从单个 paragraph 数据中提取 实体-关系-实体 三元组
    """
    triplets = []
    sentences = paragraph_data.get('sentences', [])

    for sentence in sentences:
        entities = sentence.get('entities', [])
        relations = sentence.get('relations', [])

        # 1. 建立 entity_id 到 实体详情 的映射，方便快速查找
        # 结构：{ "T1374": {"entity": "CKD", "type": "Disease"}, ... }
        entity_map = {
            e['entity_id']: {
                "name": e['entity'],
                "type": e['entity_type']
            }
            for e in entities
        }

        # 2. 遍历关系，构建三元组
        for rel in relations:
            head_id = rel.get('head_entity_id')
            tail_id = rel.get('tail_entity_id')
            rel_type = rel.get('relation_type')

            # 确保头尾实体都在当前句子的实体列表中（防止数据不一致）
            if head_id in entity_map and tail_id in entity_map:
                head_entity = entity_map[head_id]
                tail_entity = entity_map[tail_id]

                triplet = {
                    "source_file": filename,
                    "paragraph_id": paragraph_data.get('paragraph_id'),
                    "sentence": sentence.get('sentence'),
                    "head_entity": head_entity['name'],
                    "head_type": head_entity['type'],
                    "relation": rel_type,
                    "tail_entity": tail_entity['name'],
                    "tail_type": tail_entity['type']
                }
                triplets.append(triplet)

    return triplets


# 主处理流程
for filename in natsorted(os.listdir(folder_path)):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                paragraphs = data.get('paragraphs', [])

                file_triplet_count = 0
                for para in paragraphs:
                    # 提取当前段落的三元组
                    extracted = extract_triplets_from_paragraph(para, filename)
                    all_triplets.extend(extracted)
                    file_triplet_count += len(extracted)

                print(f"成功读取 {filename}，提取 {file_triplet_count} 个三元组")

            except json.JSONDecodeError:
                print(f"警告：{filename} 格式错误，已跳过")
            except KeyError as e:
                print(f"错误：{filename} 缺少关键字段 {e}，已跳过")
            except Exception as e:
                print(f"处理 {filename} 时发生未知错误：{e}")

print(f"\n所有文件处理完成，共提取 {len(all_triplets)} 个三元组")

# 预览前 3 个结果
if all_triplets:
    print("\n--- 三元组预览 ---")
    pprint.pprint(all_triplets[:3])

# 保存结果到新的 JSON 文件
output_file = 'diabetes_kg_triplets.json'
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(all_triplets, out_f, ensure_ascii=False, indent=2)
print(f"\n三元组数据已保存至：{output_file}")

'''
二、构建知识图谱
你现在的 diabetes_kg_triplets.json 文件已经包含了标准的 (头实体，关系，尾实体) 数据，这是构建图谱的核心原料。接下来的工作通常分为三个阶段：
1. 选择图谱存储方案
根据你的技术栈和需求，主要有两种主流选择：
方案 A：图数据库（推荐用于生产/复杂查询）
工具：Neo4j (最流行), Nebula Graph, TigerGraph。
优势：支持 Cypher 查询语言，擅长处理多跳查询（例如：“查找所有能治疗糖尿病且副作用包含头痛的药物”），可视化效果好。
适用场景：需要频繁查询关联关系、构建大型医疗知识库。
方案 B：内存图网络（推荐用于分析/轻量级应用）
工具：Python 库 networkx 或 pyvis。
优势：无需安装额外数据库服务，直接在 Python 脚本中运行，适合快速原型验证、计算中心度等图算法指标。
适用场景：数据分析、静态图谱生成、小规模演示。
'''
# from neo4j import GraphDatabase
#
# # 连接数据库
# uri = "bolt://localhost:7687"
# driver = GraphDatabase.driver(uri, auth=("neo4j", "your_password"))
#
# def create_graph(tx, triplet):
#     # 创建头实体节点 (带类型标签)
#     tx.run(
#         f"MERGE (h:{triplet['head_type']} {{name: $head_name}})",
#         head_name=triplet['head_entity']
#     )
#     # 创建尾实体节点
#     tx.run(
#         f"MERGE (t:{triplet['tail_type']} {{name: $tail_name}})",
#         tail_name=triplet['tail_entity']
#     )
#     # 创建关系
#     # 注意：Neo4j 的关系类型不能是变量，通常需要映射或动态拼接字符串执行
#     rel_type = triplet['relation'].replace(" ", "_")
#     tx.run(
#         f"MATCH (h:{triplet['head_type']} {{name: $head_name}}), "
#         f"(t:{triplet['tail_type']} {{name: $tail_name}}) "
#         f"MERGE (h)-[r:{rel_type}]->(t)",
#         head_name=triplet['head_entity'],
#         tail_name=triplet['tail_entity']
#     )
#
# with driver.session() as session:
#     for t in all_triplets: # 假设 all_triplets 是你之前生成的列表
#         session.execute_write(create_graph, t)


