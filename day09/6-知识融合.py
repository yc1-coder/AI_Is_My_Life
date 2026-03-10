import os
import json
from collections import defaultdict
from neo4j import GraphDatabase


# --------------------------
# 1. 知识融合逻辑 (内存级)
# --------------------------

def load_subgraphs(input_dir):
    """加载所有子图数据"""
    all_triplets = []
    if os.path.exists(input_dir):
        # 假设是合并模式的大文件
        merged_file = os.path.join(input_dir, 'all_entities_subgraphs.json')
        if os.path.exists(merged_file):
            with open(merged_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entity, triplets in data.items():
                    all_triplets.extend(triplets)
        else:
            # 独立文件模式遍历
            for fname in os.listdir(input_dir):
                if fname.endswith('_subgraph.json'):
                    with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
                        all_triplets.extend(json.load(f))
    return all_triplets


def fuse_knowledge(triplets):
    """
    执行知识融合：去重、冲突检测、置信度合并
    """
    print("🔍 开始知识融合...")

    # 1. 标准化键用于去重
    seen = set()
    fused_triplets = []
    conflict_log = []

    # 统计每条边的出现次数作为简易置信度
    edge_counts = defaultdict(int)
    edge_sources = defaultdict(list)

    for t in triplets:
        key = (t['head_entity'], t['relation'], t['tail_entity'])
        edge_counts[key] += 1
        edge_sources[key].append(t.get('source_file', 'unknown'))

    # 2. 生成融合后的列表
    for key, count in edge_counts.items():
        h, r, tail = key
        # 简单策略：保留出现次数 > 0 的边
        # 高级策略：如果存在互斥关系 (如 ADE_Drug vs Drug_Disease 同时存在且语义冲突)，在此处判断
        fused_triplets.append({
            "head_entity": h,
            "relation": r,
            "tail_entity": tail,
            "confidence": count,  # 添加置信度字段
            "sources": list(set(edge_sources[key]))
        })

    print(f"✅ 融合完成：原始 {len(triplets)} 条 -> 融合后 {len(fused_triplets)} 条")
    return fused_triplets


# --------------------------
# 2. 图谱存储 (Neo4j)
# --------------------------

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_schema(self):
        """创建索引以加速查询"""
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (r:Relation) ON (r.type)")

    def insert_triplets(self, triplets):
        """批量插入三元组"""
        with self.driver.session() as session:
            batch_size = 500
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                session.write_transaction(self._create_nodes_and_relations, batch)
                print(f"   已插入 {min(i + batch_size, len(triplets))} / {len(triplets)} 条关系")

    @staticmethod
    def _create_nodes_and_relations(tx, triplets):
        query = """
        UNWIND $triplets AS t
        MERGE (h:Entity {name: t.head_entity})
        SET h.type = COALESCE(h.type, 'Unknown')
        MERGE (tail:Entity {name: t.tail_entity})
        SET tail.type = COALESCE(tail.type, 'Unknown')
        // 动态创建关系类型需要 APOC 插件，这里使用通用关系或预定义关系映射
        // 为简化演示，我们将关系类型作为属性，或使用 apoc.create.relationship
        // 如果没有 APOC，建议使用固定关系类型如 'RELATED_TO' 并存储具体类型在属性中
        CALL apoc.create.relationship(h, t.relation, {confidence: t.confidence, sources: t.sources}, tail) YIELD rel
        RETURN count(rel)
        """
        # 注意：如果 Neo4j 未安装 APOC 插件，需改用静态关系类型或 Cypher 原生写法
        # 原生写法示例 (需预先知道所有关系类型或使用泛型):
        # 这里为了通用性，假设用户安装了 APOC，或者我们可以手动构建动态 Cypher (较复杂)
        # 替代方案：使用泛型关系 'CONNECTS'
        fallback_query = """
        UNWIND $triplets AS t
        MERGE (h:Entity {name: t.head_entity})
        MERGE (tail:Entity {name: t.tail_entity})
        MERGE (h)-[r:CONNECTS {type: t.relation, confidence: t.confidence}]->(tail)
        RETURN count(r)
        """
        tx.run(fallback_query, triplets=triplets)


def main():
    # 配置
    subgraph_dir = './all_subgraphs_result'
    output_fused_json = 'diabetes_kg_fused.json'

    # Neo4j 配置 (请根据实际情况修改)
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "q6104761"

    # 1. 加载并融合
    raw_triplets = load_subgraphs(subgraph_dir)
    if not raw_triplets:
        # 如果没有子图结果，尝试加载之前的推断文件
        if os.path.exists('diabetes_kg_inferred.json'):
            with open('diabetes_kg_inferred.json', 'r', encoding='utf-8') as f:
                raw_triplets = json.load(f)
        else:
            print("❌ 未找到任何输入数据")
            return

    fused_data = fuse_knowledge(raw_triplets)

    # 保存融合后的 JSON
    with open(output_fused_json, 'w', encoding='utf-8') as f:
        json.dump(fused_data, f, ensure_ascii=False, indent=2)
    print(f"💾 融合数据已保存至 {output_fused_json}")

    # 2. 存入 Neo4j (可选)
    try:
        print("🔗 正在连接 Neo4j...")
        neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        neo4j_conn.create_schema()
        neo4j_conn.insert_triplets(fused_data)
        neo4j_conn.close()
        print("✅ 数据已成功导入 Neo4j，请在浏览器访问 http://localhost:7474 查看图谱")
    except Exception as e:
        print(f"⚠️ Neo4j 导入失败 (可能未安装或未启动): {e}")
        print("   跳过数据库步骤，仅保留 JSON 文件。")


if __name__ == "__main__":
    main()
