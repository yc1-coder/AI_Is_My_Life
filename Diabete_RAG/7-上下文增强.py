import os
import json
import torch
import torch.nn as nn
from collections import defaultdict
import re


# --------------------------
# 1. 加载融合后的知识图谱
# --------------------------

def load_fused_kg(kg_file):
    """加载融合后的知识图谱数据"""
    print(f"📥 正在加载融合知识图谱：{kg_file}")

    if not os.path.exists(kg_file):
        raise FileNotFoundError(f"未找到融合文件：{kg_file}")

    with open(kg_file, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)

    # 构建图结构
    graph = defaultdict(list)  # head_entity -> [(relation, tail_entity), ...]
    reverse_graph = defaultdict(list)  # tail_entity -> [(relation, head_entity), ...]
    entities = set()
    relations = set()

    for triplet in kg_data:
        h = triplet['head_entity']
        r = triplet['relation']
        t = triplet['tail_entity']

        graph[h].append((r, t))
        reverse_graph[t].append((r, h))
        entities.add(h)
        entities.add(t)
        relations.add(r)

    print(f"✅ 知识图谱加载完成:")
    print(f"   实体数：{len(entities)}")
    print(f"   关系数：{len(relations)}")
    print(f"   三元组数：{len(kg_data)}")

    return kg_data, graph, reverse_graph, entities, relations


# --------------------------
# 2. 加载子图检索模型
# --------------------------

class TransE(nn.Module):
    """复用 TransE 模型进行语义相似度计算"""

    def __init__(self, num_entities, num_relations, dim=500):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.entity_embed = nn.Embedding(num_entities, dim)
        self.relation_embed = nn.Embedding(num_relations, dim)

    def get_entity_embedding(self, entity_id):
        """获取实体向量"""
        with torch.no_grad():
            return self.entity_embed(torch.tensor([entity_id], dtype=torch.long))

    def compute_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        with torch.no_grad():
            cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=1)
            return cos_sim.item()


def load_transE_model(model_path, vocab_path):
    """加载训练好的 TransE 模型"""
    print("📥 正在加载 TransE 模型...")

    # 加载词表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    id2entity_raw = vocab.get('id2entity', {})
    id2entity = {int(k): v for k, v in id2entity_raw.items()}
    entity2id_raw = vocab.get('entity2id', {})
    entity2id = {k: int(v) for k, v in entity2id_raw.items()}

    num_entities = vocab.get('num_entities', len(id2entity))
    num_relations = vocab.get('num_relations', len(vocab.get('relation2id', {})))

    # 加载模型
    model = TransE(num_entities, num_relations, dim=200)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ TransE 模型加载成功")
    else:
        print(f"⚠️ 未找到模型文件，将使用基于图的检索")
        model = None

    return model, {'entity2id': entity2id, 'id2entity': id2entity}


# --------------------------
# 3. 查询理解与实体识别
# --------------------------

def extract_entities_from_query(query, entities):
    """从查询中提取实体（简单字符串匹配，可升级为 NER 模型）"""
    found_entities = []

    # 按长度降序排列，优先匹配长实体
    sorted_entities = sorted(entities, key=len, reverse=True)

    for entity in sorted_entities:
        if entity in query:
            found_entities.append(entity)

    return found_entities


def expand_query_with_synonyms(query, graph, entities):
    """查询扩展：添加同义词、相关实体"""
    expanded_terms = []

    # 简单策略：查找实体的直接邻居
    for entity in entities:
        if entity in graph:
            for rel, tail in graph[entity]:
                expanded_terms.append(tail)

    # 去重
    expanded_terms = list(set(expanded_terms))

    return expanded_terms


# --------------------------
# 4. 子图检索与路径挖掘
# --------------------------

def retrieve_subgraph_by_entities(seed_entities, graph, reverse_graph, max_hops=2, top_k=10):
    """
    基于种子实体检索多跳子图
    """
    print(f"🔍 检索种子实体的 {max_hops}-hop 子图...")

    visited = set()
    subgraph_triplets = []
    queue = [(e, 0) for e in seed_entities]  # (entity, hop_count)

    while queue:
        current_entity, hops = queue.pop(0)

        if current_entity in visited or hops > max_hops:
            continue

        visited.add(current_entity)

        # 前向关系
        if current_entity in graph:
            for rel, tail in graph[current_entity]:
                subgraph_triplets.append({
                    'head_entity': current_entity,
                    'relation': rel,
                    'tail_entity': tail
                })
                if tail not in visited:
                    queue.append((tail, hops + 1))

        # 反向关系
        if current_entity in reverse_graph:
            for rel, head in reverse_graph[current_entity]:
                subgraph_triplets.append({
                    'head_entity': head,
                    'relation': rel,
                    'tail_entity': current_entity
                })
                if head not in visited:
                    queue.append((head, hops + 1))

    # 限制子图大小
    if len(subgraph_triplets) > top_k * len(seed_entities):
        subgraph_triplets = subgraph_triplets[:top_k * len(seed_entities)]

    print(f"   检索到 {len(subgraph_triplets)} 条三元组")
    return subgraph_triplets


def retrieve_subgraph_by_semantic(query, model, vocab, kg_data, top_k=5):
    """
    基于语义相似度检索子图（使用 TransE 向量）
    """
    if model is None:
        print("⚠️ 模型未加载，跳过语义检索")
        return []

    print(f"🔍 基于语义检索 Top-{top_k} 相关实体...")

    # 简单实现：将查询与实体名称匹配（实际应该用查询向量）
    # 这里演示用实体在 KG 中的出现频率作为权重
    entity_freq = defaultdict(int)
    for t in kg_data:
        entity_freq[t['head_entity']] += 1
        entity_freq[t['tail_entity']] += 1

    # 返回高频实体（实际应该计算查询向量与实体向量的相似度）
    sorted_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    seed_entities = [e for e, _ in sorted_entities]

    print(f"   检索到实体：{seed_entities}")
    return seed_entities


# --------------------------
# 5. 上下文模板生成
# --------------------------

def triplets_to_natural_language(triplets):
    """将三元组转换为自然语言描述"""
    descriptions = []

    for t in triplets:
        h = t['head_entity']
        r = t['relation']
        tail = t['tail_entity']

        # 关系类型映射到自然语言
        relation_templates = {
            'ADE_Drug': f"{h} 可能导致不良反应 {tail}",
            'Drug_Disease': f"{h} 可用于治疗 {tail}",
            'Pathogenesis_Disease': f"{h} 是 {tail} 的发病机制",
            'Symptom_Disease': f"{h} 是 {tail} 的症状",
            'Class_Drug': f"{h} 属于药物类别 {tail}",
            'IND_Disease': f"{h} 的适应症包括 {tail}",
            'RELATED_TO': f"{h} 与 {tail} 相关 ({r})"
        }

        template = relation_templates.get(r, f"{h} {r} {tail}")
        descriptions.append(template)

    return descriptions


def build_rag_context(query, retrieved_subgraph, confidence_threshold=0.5):
    """
    构建 RAG 上下文 Prompt
    """
    print("\n📄 构建 RAG 上下文...")

    # 1. 过滤高置信度三元组
    filtered_triplets = [
        t for t in retrieved_subgraph
        if t.get('confidence', 1.0) >= confidence_threshold
    ]

    # 2. 转换为自然语言
    kg_sentences = triplets_to_natural_language(filtered_triplets)

    # 3. 构建结构化上下文
    context_parts = []

    # 系统指令
    system_instruction = """你是一位专业的医疗知识助手。请基于以下提供的医学知识图谱信息，准确、清晰地回答用户的问题。
如果图谱信息与问题不相关，请诚实地说明。不要编造信息。"""

    context_parts.append(system_instruction)

    # 知识图谱上下文
    if kg_sentences:
        kg_context = "\n【知识图谱信息】\n" + "\n".join([f"• {s}" for s in kg_sentences])
        context_parts.append(kg_context)
    else:
        context_parts.append("\n【知识图谱信息】\n未找到相关知识。")

    # 用户问题
    user_query = f"\n【用户问题】\n{query}"
    context_parts.append(user_query)

    # 完整 Prompt
    full_prompt = "\n".join(context_parts)

    print(f"   上下文长度：{len(full_prompt)} 字符")
    print(f"   包含知识条数：{len(kg_sentences)}")

    return full_prompt, kg_sentences


# --------------------------
# 6. Ollama 集成与问答
# --------------------------

def query_ollama(prompt, model_name="llama3:8b-instruct-q4_K_M"):
    """调用 Ollama 模型生成回答"""
    try:
        import ollama

        print(f"🤖 正在调用 Ollama ({model_name})...")

        client = ollama.Client(host="http://localhost:11434")

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = client.chat(
            model=model_name,
            messages=messages
        )

        ai_response = response['message']['content']
        print(f"✅ 生成成功")

        return ai_response

    except ImportError:
        print("⚠️ 未安装 ollama 库：pip install ollama")
        return None
    except Exception as e:
        print(f"❌ 调用 Ollama 失败：{e}")
        return None


# --------------------------
# 7. 完整 RAG 流程
# --------------------------

def rag_pipeline(query, kg_file='diabetes_kg_fused.json',
                 model_path='./transE_model.pth',
                 vocab_path='./kg_embedding_data/vocab.json'):
    """
    完整的 RAG 问答流程
    """
    print("=" * 60)
    print(f"🔍 RAG 问答流程启动")
    print(f"   问题：{query}")
    print("=" * 60)

    # Step 1: 加载知识图谱
    kg_data, graph, reverse_graph, entities, relations = load_fused_kg(kg_file)

    # Step 2: 加载模型
    model, vocab = load_transE_model(model_path, vocab_path)

    # Step 3: 实体识别
    seed_entities = extract_entities_from_query(query, entities)
    print(f"\n🎯 识别到的实体：{seed_entities}")

    if not seed_entities:
        print("⚠️ 未识别到知识库中的实体，尝试语义检索...")
        seed_entities = retrieve_subgraph_by_semantic(query, model, vocab, kg_data, top_k=5)

    if not seed_entities:
        print("❌ 无法找到相关实体，将使用通用回答")

    # Step 4: 子图检索
    retrieved_subgraph = retrieve_subgraph_by_entities(
        seed_entities, graph, reverse_graph, max_hops=2, top_k=10
    )

    # Step 5: 构建 RAG 上下文
    rag_context, kg_info = build_rag_context(query, retrieved_subgraph)

    # Step 6: 调用 LLM 生成回答
    print("\n" + "=" * 60)
    ai_answer = query_ollama(rag_context)

    if ai_answer:
        print("\n" + "=" * 60)
        print("💬 AI 回答:")
        print(ai_answer)
        print("=" * 60)
    else:
        # 降级策略：直接返回知识图谱信息
        print("\n" + "=" * 60)
        print("⚠️ LLM 不可用，直接展示知识图谱信息:")
        for sentence in kg_info:
            print(f"  • {sentence}")
        print("=" * 60)

    return {
        'query': query,
        'entities': seed_entities,
        'subgraph': retrieved_subgraph,
        'context': rag_context,
        'answer': ai_answer
    }


# --------------------------
# 8. 交互式演示
# --------------------------

def interactive_demo():
    """交互式问答演示"""
    print("\n" + "=" * 60)
    print("🎓 知识图谱 RAG 交互式问答系统")
    print("=" * 60)
    print("提示：输入问题后按回车，输入 'quit' 退出\n")

    while True:
        query = input("请输入你的问题：").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 感谢使用，再见!")
            break

        if not query:
            continue

        try:
            result = rag_pipeline(query)

            # 可选：保存结果
            save_result = input("\n是否保存本次问答结果？(y/n): ").strip().lower()
            if save_result == 'y':
                with open('rag_qa_log.json', 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print("💾 结果已保存至 rag_qa_log.json")

        except Exception as e:
            print(f"❌ 处理失败：{e}")
            import traceback
            traceback.print_exc()


# --------------------------
# 9. 主函数
# --------------------------

def main():
    # 示例问题列表
    demo_queries = [
        "格列本脲有什么不良反应？",
        "2 型糖尿病的治疗药物有哪些？",
        "胰岛素促泌剂的作用机制是什么？",
        "二甲双胍可以用于治疗什么疾病？"
    ]

    print("=" * 60)
    print("🚀 上下文增强模块 - 知识图谱 RAG 系统")
    print("=" * 60)
    print("\n请选择运行模式:")
    print("1. 运行预设示例问题")
    print("2. 进入交互式问答")
    print("3. 自定义单个问题测试")

    choice = input("\n请输入选项 (1/2/3): ").strip()

    if choice == '1':
        # 批量测试示例问题
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'=' * 60}")
            print(f"问题 {i}: {query}")
            print('=' * 60)
            rag_pipeline(query)
            input("\n按回车继续下一个问题...")

    elif choice == '2':
        # 交互模式
        interactive_demo()

    elif choice == '3':
        # 单题测试
        query = input("请输入问题：").strip()
        rag_pipeline(query)

    else:
        print("❌ 无效选项")


if __name__ == "__main__":
    main()
