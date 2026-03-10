import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


# --------------------------
# 1. 数据预处理模块
# --------------------------

class KGDataset(Dataset):
    """自定义数据集，用于加载三元组"""

    def __init__(self, triplets, entity2id, relation2id):
        self.triplets = triplets
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        h = self.entity2id.get(t['head_entity'])
        r = self.relation2id.get(t['relation'])
        t_id = self.entity2id.get(t['tail_entity'])

        if h is None or r is None or t_id is None:
            return self.__getitem__((idx + 1) % len(self))

        # 确保返回的是 LongTensor (索引类型)
        return torch.tensor(h, dtype=torch.long), torch.tensor(r, dtype=torch.long), torch.tensor(t_id,
                                                                                                  dtype=torch.long)


def prepare_data(input_file, output_dir='./data', split_ratio=(0.8, 0.1, 0.1)):
    """
    加载 JSON 数据，构建词表，划分数据集并保存为 txt 格式
    """
    print(f"📥 正在加载 {input_file} ...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到文件：{input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        triplets = json.load(f)

    # 1. 构建实体和关系映射
    entities = set()
    relations = set()
    for t in triplets:
        entities.add(t['head_entity'])
        entities.add(t['tail_entity'])
        relations.add(t['relation'])

    entity2id = {e: i for i, e in enumerate(sorted(list(entities)))}
    id2entity = {i: e for e, i in entity2id.items()}
    relation2id = {r: i for i, r in enumerate(sorted(list(relations)))}
    id2relation = {i: r for r, i in relation2id.items()}

    num_entities = len(entities)
    num_relations = len(relations)

    print(f"   实体数量：{num_entities}, 关系数量：{num_relations}")

    # 2. 打乱并划分数据集
    random.shuffle(triplets)
    n = len(triplets)
    train_end = int(n * split_ratio[0])
    valid_end = train_end + int(n * split_ratio[1])

    train_data = triplets[:train_end]
    valid_data = triplets[train_end:valid_end]
    test_data = triplets[valid_end:]

    # 3. 保存映射文件
    os.makedirs(output_dir, exist_ok=True)

    def save_mapping(data, filename):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for t in data:
                h = entity2id.get(t['head_entity'])
                r = relation2id.get(t['relation'])
                tail = entity2id.get(t['tail_entity'])
                if h is not None and r is not None and tail is not None:
                    f.write(f"{t['head_entity']}\t{t['relation']}\t{t['tail_entity']}\n")
        return path

    train_path = save_mapping(train_data, 'train.txt')
    valid_path = save_mapping(valid_data, 'valid.txt')
    test_path = save_mapping(test_data, 'test.txt')

    # 保存词表
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'entity2id': entity2id,
            'id2entity': id2entity,
            'relation2id': relation2id,
            'id2relation': id2relation,
            'num_entities': num_entities,
            'num_relations': num_relations
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 数据预处理完成:")
    print(f"   训练集：{len(train_data)} 条 -> {train_path}")
    print(f"   验证集：{len(valid_data)} 条 -> {valid_path}")
    print(f"   测试集：{len(test_data)} 条 -> {test_path}")

    return train_data, valid_data, test_data, entity2id, relation2id, id2entity, num_entities, num_relations


# --------------------------
# 2. 模型定义 (TransE)
# --------------------------

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=500):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.entity_embed = nn.Embedding(num_entities, dim)
        self.relation_embed = nn.Embedding(num_relations, dim)

        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

    def forward(self, h, r, t):
        """
        计算得分：score = || h + r - t ||
        输入 h, r, t 均为 [batch_size] 的索引向量
        返回 [batch_size] 的得分向量
        """
        h_emb = self.entity_embed(h)  # [batch, dim]
        r_emb = self.relation_embed(r)  # [batch, dim]
        t_emb = self.entity_embed(t)  # [batch, dim]

        # 确保维度一致
        score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
        return score

    def predict_tail(self, h, r, top_k=5):
        """
        链接预测
        """
        self.eval()
        with torch.no_grad():
            h_emb = self.entity_embed(torch.tensor([h], dtype=torch.long))
            r_emb = self.relation_embed(torch.tensor([r], dtype=torch.long))
            target_vec = h_emb + r_emb

            all_entities = torch.arange(self.num_entities, dtype=torch.long)
            all_emb = self.entity_embed(all_entities)

            # 计算距离: [1, dim] vs [N, dim] -> [N]
            dists = torch.norm(target_vec - all_emb, p=2, dim=1)

            scores, indices = torch.topk(dists, k=top_k, largest=False)
            return indices.tolist(), scores.tolist()


# --------------------------
# 3. 训练逻辑 (修复版)
# --------------------------

def train_model(train_data, num_entities, num_relations, entity2id, relation2id,
                dim=500, epochs=100, lr=0.01, batch_size=512, neg_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 开始在设备 {device} 上训练 TransE 模型...")

    dataset = KGDataset(train_data, entity2id, relation2id)
    # drop_last=True 防止最后一批数据过小导致广播问题（可选，但推荐）
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TransE(num_entities, num_relations, dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        for h_batch, r_batch, t_batch in loader:
            h_batch = h_batch.to(device)
            r_batch = r_batch.to(device)
            t_batch = t_batch.to(device)

            optimizer.zero_grad()

            # 1. 正样本得分 [batch_size]
            pos_score = model(h_batch, r_batch, t_batch)

            # 2. 负采样
            # 生成 [batch_size, neg_samples] 的随机尾实体索引
            neg_t_batch = torch.randint(0, num_entities, (h_batch.size(0), neg_samples), dtype=torch.long).to(device)

            neg_loss = 0
            margin = 1.0

            # 逐个负样本计算损失
            for i in range(neg_samples):
                neg_t = neg_t_batch[:, i]  # [batch_size]

                # 计算负样本得分 [batch_size]
                neg_score = model(h_batch, r_batch, neg_t)

                # 确保形状完全一致再计算 Loss
                # pos_score: [batch], neg_score: [batch]
                loss_item = torch.relu(margin + pos_score - neg_score)
                neg_loss += loss_item.mean()

            loss = neg_loss / neg_samples
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / batch_count
            print(f"   Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    print("✅ 模型训练完成")
    return model


# --------------------------
# 4. 推理与演示
# --------------------------

def run_link_prediction(model, id2entity, entity2id, relation2id):
    """
    演示链接预测功能
    """
    print("\n🔍 开始进行链接预测演示...")
    model.eval()

    test_cases = [
        ("格列本脲", "ADE_Drug"),
        ("2型糖尿病", "Pathogenesis_Disease"),
        ("胰岛素促泌剂", "Drug_Disease")
    ]

    for head_name, rel_name in test_cases:
        if head_name not in entity2id or rel_name not in relation2id:
            print(f"   ⚠️ 跳过未知实体/关系：{head_name} - {rel_name}")
            continue

        h_id = entity2id[head_name]
        r_id = relation2id[rel_name]

        top_ids, scores = model.predict_tail(h_id, r_id, top_k=5)

        print(f"\n   查询：({head_name}, {rel_name}, ?)")
        print(f"   预测结果 (尾实体 | 距离分数):")
        for idx, score in zip(top_ids, scores):
            entity_name = id2entity.get(idx, "Unknown")
            print(f"      - {entity_name} (Score: {score:.4f})")


# --------------------------
# 5. 主执行流程
# --------------------------

def main():
    input_file = 'diabetes_kg_inferred.json'
    output_dir = './kg_embedding_data'
    model_save_path = './transE_model.pth'

    if not os.path.exists(input_file):
        print(f"❌ 错误：未找到 {input_file}，请先运行前序步骤生成该文件。")
        return

    train_data, _, _, entity2id, relation2id, id2entity, num_entities, num_relations = prepare_data(input_file,
                                                                                                    output_dir)

    model = train_model(
        train_data=train_data,
        num_entities=num_entities,
        num_relations=num_relations,
        entity2id=entity2id,
        relation2id=relation2id,
        dim=200,
        epochs=50,
        lr=0.01,
        batch_size=128,
        neg_samples=5
    )

    torch.save(model.state_dict(), model_save_path)
    print(f"💾 模型权重已保存至：{model_save_path}")

    run_link_prediction(model, id2entity, entity2id, relation2id)


if __name__ == "__main__":
    main()
