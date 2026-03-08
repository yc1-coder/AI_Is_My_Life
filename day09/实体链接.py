import os
import json


def load_synonyms(filepath):
    """加载同义词字典"""
    if not os.path.exists(filepath):
        print(f"警告：未找到 {filepath}，将跳过实体链接步骤。")
        return {}

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_entity(entity_name, synonym_map):
    """
    将实体名映射为标准名。
    逻辑：如果实体名在某个标准名的同义词列表中，则返回该标准名；否则返回原名。
    """
    for standard_name, synonyms in synonym_map.items():
        if entity_name == standard_name or entity_name in synonyms:
            return standard_name
    return entity_name


def run_entity_linking():
    input_file = 'diabetes_kg_triplets.json'
    synonym_file = 'synonym_dict.json'
    output_file = 'diabetes_kg_linked.json'

    # 1. 加载数据
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"请先运行 read_json.py 的前置步骤生成 {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        triplets = json.load(f)

    synonym_map = load_synonyms(synonym_file)

    linked_triplets = []
    seen_relations = set()
    stats = {"original": len(triplets), "merged": 0}

    # 2. 处理每个三元组
    for t in triplets:
        # 标准化头尾实体
        head_std = normalize_entity(t['head_entity'], synonym_map)
        tail_std = normalize_entity(t['tail_entity'], synonym_map)

        # 更新三元组中的实体名称
        t['head_entity'] = head_std
        t['tail_entity'] = tail_std

        # 3. 去重逻辑：基于 (标准头实体，关系，标准尾实体) 进行去重
        relation_key = (head_std, t['relation'], tail_std)

        if relation_key not in seen_relations:
            seen_relations.add(relation_key)
            linked_triplets.append(t)
        else:
            stats["merged"] += 1

    # 4. 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(linked_triplets, f, ensure_ascii=False, indent=2)

    print(f"✅ 实体链接完成！")
    print(f"   原始三元组：{stats['original']} 条")
    print(f"   合并重复项：{stats['merged']} 条")
    print(f"   最终三元组：{len(linked_triplets)} 条")
    print(f"   结果已保存至：{output_file}")


if __name__ == "__main__":
    run_entity_linking()
