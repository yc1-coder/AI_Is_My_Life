import os
import json
from collections import defaultdict


def load_kg(filepath):
    """加载知识图谱三元组"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到文件：{filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_kg(triplets, filepath):
    """保存知识图谱"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"✅ 推理结果已保存至：{filepath}")


def build_index(triplets):
    """
    构建索引以加速查询
    返回:
      entity_relations: {entity: [(relation, other_entity, type)]}
      subclass_map: {subclass: parent_class} (假设存在 Class_Disease 或类似的层级关系)
    """
    # 用于存储实体相关的关系
    entity_graph = defaultdict(list)
    # 用于存储子类 -> 父类 的映射 (根据数据中的 Class_Disease 或 Drug 层级推断)
    # 注意：当前数据中层级关系不明显，这里主要演示基于路径的推理逻辑
    return entity_graph


def infer_transitive_relations(triplets):
    """
    规则推理 1: 传递性推理
    逻辑示例：如果 A -[导致]-> B 且 B -[属于]-> C，则推导 A -[关联]-> C
    在本医疗场景中，我们尝试推导：疾病 -> 并发症 -> 具体症状/器官
    """
    new_triplets = []

    # 构建快速查找表: {head: [(rel, tail)]}
    head_to_tails = defaultdict(list)
    for t in triplets:
        head_to_tails[t['head_entity']].append((t['relation'], t['tail_entity'], t))

    # 定义推理规则集合
    # 规则：Pathogenesis_Disease (病因->病) + Anatomy_Disease (解剖->病) 不直接传递，但可用于发现共病
    # 这里演示一个具体的医疗逻辑：
    # 如果 Drug -[ADE_Drug]-> Symptom (副作用), 且 Symptom 是某种 Disease 的表现，可建立间接联系
    # 由于当前数据主要是 (Entity, Relation, Disease) 结构，我们重点做 "子类继承" 或 "并发症链" 的模拟

    seen_keys = set()

    # 模拟推理：基于现有的 "Anatomy_Disease" 和 "Pathogenesis_Disease" 丰富 "Disease_Disease" 关系
    # 例如：如果 "血管" -> "血管病变", "血管病变" -> "2型糖尿病" (反向),
    # 我们可以尝试寻找中间节点。

    # 这里做一个具体的简单推理示例：
    # 如果 A (Drug) -> B (Disease) 且 B (Disease) -> C (Complication/Disease via some logic)
    # 由于当前数据尾实体多为 Disease，头实体多样，我们尝试推导 Drug 到 Complication 的直接关系

    # 假设逻辑：如果 Drug 治疗 Disease X，且 Disease X 会导致 Complication Y (需要外部知识或更多三元组)
    # 在当前数据集中，我们主要做去重后的置信度增强或简单的层级补全

    # 示例：如果发现 "格列本脲" -> "2型糖尿病" 和 "磺脲类促泌剂" -> "2型糖尿病"
    # 且我们知道 "格列本脲" 是 "磺脲类" 的一种 (需额外字典，此处暂略)，则可互相验证。

    # --- 实际可执行的推理逻辑：基于共同实体的关联发现 ---
    # 找出所有指向同一疾病的病因或解剖结构，建立它们之间的潜在关联
    disease_sources = defaultdict(list)
    for t in triplets:
        if t['tail_type'] == 'Disease':
            disease_sources[t['tail_entity']].append(t)

    # 规则：如果 病因A -> 疾病X 且 病因B -> 疾病X，则 病因A 与 病因B 可能存在协同或并发关系
    # 这里我们仅记录日志，不盲目添加新边，以免噪声过大
    # 真正的推理通常需要更复杂的图算法或嵌入模型

    # --- 演示：添加一条基于领域知识的硬编码推理 (模拟) ---
    # 假设：所有 "磺脲类促泌剂" 的副作用 "低血糖" 也适用于其子类 (如果数据中有子类关系)
    # 由于数据中缺乏明确的 is_a 关系，我们暂时不做复杂的传递闭包，
    # 而是专注于清洗和标记高价值路径。

    print("🔍 开始执行规则推理...")

    # 示例：推导 "糖尿病" 与 "并发症" 的更强连接
    # 遍历所有 Anatomy_Disease 关系，如果尾实体是并发症，头实体是器官
    # 我们可以反推：该器官的疾病风险与糖尿病强相关

    inferred_count = 0

    # 这里为了演示，我们创建一个虚拟的推理结果
    # 在实际生产中，这里会运行图算法 (如 PageRank, Random Walk) 或 规则引擎
    for t in triplets:
        # 示例规则：如果某药物导致低血糖 (ADE)，且该药物用于治疗糖尿病
        # 我们可以显式地标记 "糖尿病治疗" 与 "低血糖风险" 的关联 (虽然这是常识，但在图谱中显式化很有用)
        if t['relation'] == 'ADE_Drug' and t['head_entity'] == '低血糖':
            drug = t['tail_entity']
            # 查找该药物治疗的疾病
            for other_t in triplets:
                if other_t['head_entity'] == drug and other_t['relation'] == 'Drug_Disease':
                    disease = other_t['tail_entity']
                    # 构造新关系：疾病 -> 风险因素 -> 低血糖 (反向推理)
                    # 或者：药物 -> 治疗疾病 -> 伴随风险
                    pass  # 此处逻辑可根据需求扩展

    # 由于纯规则推理依赖具体的本体定义，以下返回原始数据加上可能的推理扩展
    # 在实际项目中，你会在这里 append 新的 triplet 到 new_triplets
    return new_triplets


def run_inference():
    input_file = 'diabetes_kg_linked.json'
    output_file = 'diabetes_kg_inferred.json'

    print(f"📥 正在加载 {input_file} ...")
    triplets = load_kg(input_file)
    original_count = len(triplets)

    # 执行推理
    inferred_triplets = infer_transitive_relations(triplets)

    # 合并结果 (去重)
    all_triplets = triplets + inferred_triplets
    seen = set()
    final_triplets = []
    for t in all_triplets:
        key = (t['head_entity'], t['relation'], t['tail_entity'])
        if key not in seen:
            seen.add(key)
            final_triplets.append(t)

    stats = {
        "original": original_count,
        "inferred_new": len(inferred_triplets),
        "final_total": len(final_triplets)
    }

    save_kg(final_triplets, output_file)

    print(f"📊 推理统计:")
    print(f"   原始三元组：{stats['original']}")
    print(f"   新增推理三元组：{stats['inferred_new']}")
    print(f"   最终三元组总数：{stats['final_total']}")


if __name__ == "__main__":
    run_inference()
