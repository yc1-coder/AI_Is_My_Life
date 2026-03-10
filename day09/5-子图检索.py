import os
import json
import torch
import torch.nn as nn
import time
from tqdm import tqdm


# 复用之前的 TransE 模型定义
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=500):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.entity_embed = nn.Embedding(num_entities, dim)
        self.relation_embed = nn.Embedding(num_relations, dim)


def load_resources(model_path, vocab_path, kg_path):
    """加载模型、词表和原始图谱数据"""
    print("📥 正在加载资源...")

    # 1. 加载词表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_raw = json.load(f)

    # 【关键修复】JSON 加载后，数字键会变成字符串 (如 "0", "1")
    # 我们需要将 id2entity 和 entity2id 的键统一处理，防止 KeyError
    # 确保 id2entity 的键是整数
    id2entity_raw = vocab_raw.get('id2entity', {})
    id2entity = {int(k): v for k, v in id2entity_raw.items()}

    # 确保 entity2id 的值是整数 (通常已经是，但做个保险)
    entity2id_raw = vocab_raw.get('entity2id', {})
    entity2id = {k: int(v) for k, v in entity2id_raw.items()}

    # 更新 vocab 字典以便后续使用
    vocab = {
        'entity2id': entity2id,
        'id2entity': id2entity,
        'relation2id': vocab_raw.get('relation2id', {}),
        'id2relation': vocab_raw.get('id2relation', {}),
        'num_entities': vocab_raw.get('num_entities', len(id2entity)),
        'num_relations': vocab_raw.get('num_relations', len(vocab_raw.get('relation2id', {})))
    }

    # 2. 加载模型
    model = TransE(vocab['num_entities'], vocab['num_relations'], dim=200)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 3. 加载原始三元组
    with open(kg_path, 'r', encoding='utf-8') as f:
        triplets = json.load(f)

    # 确保 triplets 是列表
    if isinstance(triplets, dict):
        # 如果意外加载为字典，尝试提取值
        triplets = list(triplets.values())

    print(f"✅ 资源加载完成：{len(vocab['id2entity'])} 个实体，{len(triplets)} 条三元组")
    return model, vocab, triplets


def generate_all_subgraphs(model, vocab, triplets, output_dir='./all_subgraphs', mode='merged', top_k=15):
    """
    生成所有实体的子图
    """
    os.makedirs(output_dir, exist_ok=True)

    entity2id = vocab['entity2id']
    id2entity = vocab['id2entity']
    num_entities = vocab['num_entities']

    # 预提取所有实体向量以加速计算
    with torch.no_grad():
        all_embeddings = model.entity_embed.weight.cpu()

    all_subgraphs_data = {}
    start_time = time.time()

    print(f"🚀 开始生成全量子图 (共 {num_entities} 个实体)...")

    # 使用 tqdm 显示进度
    # 【关键修复】确保遍历的是整数范围，且访问 id2entity 时使用整数键
    for idx in tqdm(range(num_entities), desc="处理实体"):
        # 安全检查：如果 idx 不在 id2entity 中，跳过
        if idx not in id2entity:
            continue

        seed_name = id2entity[idx]
        seed_id = idx

        # 1. 获取种子向量
        seed_vec = all_embeddings[seed_id].unsqueeze(0)  # [1, dim]

        # 2. 计算距离并获取 Top-K 邻居
        dists = torch.norm(seed_vec - all_embeddings, p=2, dim=1)
        k = min(top_k + 1, num_entities)
        scores, indices = torch.topk(dists, k=k, largest=False)

        neighbor_ids = indices.tolist()

        # 移除自己
        if neighbor_ids and neighbor_ids[0] == seed_id:
            neighbor_ids = neighbor_ids[1:]
        else:
            if seed_id in neighbor_ids:
                neighbor_ids.remove(seed_id)

        # 3. 基于邻居 ID 从原始三元组中提取边
        neighbor_set = set(neighbor_ids)
        neighbor_set.add(seed_id)

        subgraph_triplets = []
        for t in triplets:
            # 安全获取 ID，防止三元组中包含未登录词
            h_id = entity2id.get(t.get('head_entity'))
            t_id = entity2id.get(t.get('tail_entity'))

            if h_id is not None and t_id is not None:
                if h_id in neighbor_set and t_id in neighbor_set:
                    subgraph_triplets.append(t)

        # 4. 存储结果
        if mode == 'merged':
            all_subgraphs_data[seed_name] = subgraph_triplets
        elif mode == 'separate':
            safe_name = "".join([c for c in seed_name if c.isalnum() or c in ('_', '-')])
            if not safe_name:
                safe_name = f"entity_{seed_id}"
            file_path = os.path.join(output_dir, f"{safe_name}_subgraph.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(subgraph_triplets, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    duration = end_time - start_time

    # 保存合并文件
    if mode == 'merged':
        output_file = os.path.join(output_dir, 'all_entities_subgraphs.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_subgraphs_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 全量子图生成完成 (合并模式)")
        print(f"   总耗时：{duration:.2f} 秒")
        print(f"   结果已保存至：{output_file}")
        print(f"   包含实体数：{len(all_subgraphs_data)}")

        total_edges = sum(len(v) for v in all_subgraphs_data.values())
        avg_edges = total_edges / len(all_subgraphs_data) if all_subgraphs_data else 0
        print(f"   平均每个子图包含边数：{avg_edges:.2f}")

    elif mode == 'separate':
        print(f"\n✅ 全量子图生成完成 (独立文件模式)")
        print(f"   总耗时：{duration:.2f} 秒")
        print(f"   结果已保存至目录：{output_dir}")


def main():
    model_path = './transE_model.pth'
    vocab_path = './kg_embedding_data/vocab.json'
    kg_path = 'diabetes_kg_inferred.json'
    output_dir = './all_subgraphs_result'

    if not os.path.exists(model_path):
        print("❌ 错误：未找到模型文件，请先运行 4-图嵌入学习.py")
        return
    if not os.path.exists(vocab_path):
        print("❌ 错误：未找到词表文件")
        return
    if not os.path.exists(kg_path):
        print("❌ 错误：未找到图谱数据文件")
        return

    try:
        model, vocab, triplets = load_resources(model_path, vocab_path, kg_path)

        generate_all_subgraphs(
            model=model,
            vocab=vocab,
            triplets=triplets,
            output_dir=output_dir,
            mode='merged',
            top_k=15
        )

    except Exception as e:
        print(f"❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
