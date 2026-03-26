import streamlit as st
import os
import json
import torch
import torch.nn as nn
from collections import defaultdict
import time

# 页面配置
st.set_page_config(
    page_title="糖尿病知识图谱 RAG 系统",
    page_icon="🏥",
    layout="wide"
)

# 自定义 CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E88E5;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=500):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.entity_embed = nn.Embedding(num_entities, dim)
        self.relation_embed = nn.Embedding(num_relations, dim)


@st.cache_resource
def load_fused_kg(kg_file):
    if not os.path.exists(kg_file):
        return None, None, None, None, None

    with open(kg_file, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)

    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
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

    return kg_data, graph, reverse_graph, entities, relations


@st.cache_resource
def load_transE_model(model_path, vocab_path):
    if not os.path.exists(vocab_path):
        return None, None

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    id2entity_raw = vocab.get('id2entity', {})
    id2entity = {int(k): v for k, v in id2entity_raw.items()}
    entity2id_raw = vocab.get('entity2id', {})
    entity2id = {k: int(v) for k, v in entity2id_raw.items()}

    num_entities = vocab.get('num_entities', len(id2entity))
    num_relations = vocab.get('num_relations', len(vocab.get('relation2id', {})))

    model = TransE(num_entities, num_relations, dim=200)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            model_loaded = True
        except:
            model_loaded = False
    else:
        model_loaded = False

    return model if model_loaded else None, {'entity2id': entity2id, 'id2entity': id2entity}


def extract_entities_from_query(query, entities):
    found_entities = []
    sorted_entities = sorted(entities, key=len, reverse=True)

    for entity in sorted_entities:
        if entity in query:
            found_entities.append(entity)

    return found_entities


def retrieve_subgraph_by_entities(seed_entities, graph, reverse_graph, max_hops=2, top_k=10):
    visited = set()
    subgraph_triplets = []
    queue = [(e, 0) for e in seed_entities]

    while queue:
        current_entity, hops = queue.pop(0)

        if current_entity in visited or hops > max_hops:
            continue

        visited.add(current_entity)

        if current_entity in graph:
            for rel, tail in graph[current_entity]:
                subgraph_triplets.append({
                    'head_entity': current_entity,
                    'relation': rel,
                    'tail_entity': tail
                })
                if tail not in visited:
                    queue.append((tail, hops + 1))

        if current_entity in reverse_graph:
            for rel, head in reverse_graph[current_entity]:
                subgraph_triplets.append({
                    'head_entity': head,
                    'relation': rel,
                    'tail_entity': current_entity
                })
                if head not in visited:
                    queue.append((head, hops + 1))

    if len(subgraph_triplets) > top_k * len(seed_entities):
        subgraph_triplets = subgraph_triplets[:top_k * len(seed_entities)]

    return subgraph_triplets


def triplets_to_natural_language(triplets):
    descriptions = []

    relation_templates = {
        'ADE_Drug': "{} 可能导致不良反应 {}",
        'Drug_Disease': "{} 可用于治疗 {}",
        'Pathogenesis_Disease': "{} 是 {} 的发病机制",
        'Symptom_Disease': "{} 是 {} 的症状",
        'Class_Drug': "{} 属于药物类别 {}",
        'IND_Disease': "{} 的适应症包括 {}",
        'RELATED_TO': "{} 与 {} 相关"
    }

    for t in triplets:
        h = t['head_entity']
        r = t['relation']
        tail = t['tail_entity']

        if r in relation_templates:
            template = relation_templates[r]
            desc = template.format(h, tail)
        else:
            desc = f"{h} {r} {tail}"

        descriptions.append(desc)

    return descriptions


def build_rag_context(query, retrieved_subgraph, confidence_threshold=0.5):
    filtered_triplets = [
        t for t in retrieved_subgraph
        if t.get('confidence', 1.0) >= confidence_threshold
    ]

    kg_sentences = triplets_to_natural_language(filtered_triplets)

    system_instruction = """你是一位专业的医疗知识助手。请基于以下提供的医学知识图谱信息回答用户问题。
如果图谱信息与问题不相关，请诚实地说明。请用中文回答。"""

    context_parts = [system_instruction]

    if kg_sentences:
        kg_context = "\n【知识图谱信息】\n" + "\n".join([f"• {s}" for s in kg_sentences])
        context_parts.append(kg_context)
    else:
        context_parts.append("\n【知识图谱信息】\n未找到相关知识。")

    user_query = f"\n【用户问题】\n{query}"
    context_parts.append(user_query)

    full_prompt = "\n".join(context_parts)

    return full_prompt, kg_sentences


def query_ollama(prompt, model_name="llama3:8b-instruct-q4_K_M"):
    try:
        import ollama

        client = ollama.Client(host="http://localhost:11434")

        messages = [
            {"role": "system", "content": "请用中文回答。"},
            {"role": "user", "content": prompt}
        ]

        response = client.chat(model=model_name, messages=messages)
        ai_response = response['message']['content']

        return ai_response
    except:
        return None


# 主界面
st.markdown('<h1 class="main-header">🏥 糖尿病知识图谱 RAG 问答系统</h1>', unsafe_allow_html=True)
st.divider()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置")

    kg_file = st.text_input("知识图谱文件", value="diabetes_kg_fused.json")
    model_path = st.text_input("模型文件", value="./transE_model.pth")
    vocab_path = st.text_input("词表文件", value="./kg_embedding_data/vocab.json")

    st.divider()

    max_hops = st.slider("最大检索跳数", 1, 3, 2)
    top_k = st.slider("Top-K 邻居数", 5, 20, 10)
    confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.5, step=0.1)

# 加载数据
if os.path.exists(kg_file):
    kg_data, graph, reverse_graph, entities, relations = load_fused_kg(kg_file)

    if kg_data:
        # 统计卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 实体总数", len(entities))
        with col2:
            st.metric("🔗 关系总数", len(relations))
        with col3:
            st.metric("📝 三元组总数", len(kg_data))
        with col4:
            model, vocab = load_transE_model(model_path, vocab_path)
            st.metric("🤖 语义模型", "已加载" if model else "未加载")

        st.divider()

        # 对话历史
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "context" in msg:
                    with st.expander("📖 查看检索知识"):
                        st.markdown(msg["context"])

        # 聊天输入
        if prompt := st.chat_input("请输入问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    start = time.time()

                    seed_entities = extract_entities_from_query(prompt, entities)
                    retrieved = retrieve_subgraph_by_entities(
                        seed_entities, graph, reverse_graph, max_hops, top_k
                    )
                    rag_ctx, kg_sents = build_rag_context(prompt, retrieved, confidence_threshold)
                    answer = query_ollama(rag_ctx)

                    resp_time = time.time() - start

                    if answer:
                        st.markdown(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "context": rag_ctx,
                            "entities": seed_entities,
                            "triplets": retrieved
                        })
                        st.caption(f"⏱️ {resp_time:.2f}s | 🎯 {len(seed_entities)}实体 | 📊 {len(retrieved)}三元组")
                    else:
                        if kg_sents:
                            st.markdown("⚠️ LLM 不可用，以下是 KG 信息:")
                            for s in kg_sents:
                                st.markdown(f"• {s}")
                        else:
                            st.markdown("❌ 未找到相关知识")

        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
else:
    st.error(f"❌ 文件不存在：{kg_file}")
    st.info("💡 请先运行前面的脚本生成知识图谱文件")
