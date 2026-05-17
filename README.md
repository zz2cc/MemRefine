# Iterative Memory Optimization via Self-Comparison

无需重新训练 LLM，通过多轮自我对比迭代优化对话记忆文档生成质量。

借鉴 3DrawAgent 的"对比经验提取"范式：生成候选 → 评分排序 → 裁判对比高低分 → 提取写作规则 → 注入下一轮 prompt。

## 快速开始

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 下载模型

```bash
git clone https://gitee.com/hf-models/all-MiniLM-L6-v2.git local_models/all-MiniLM-L6-v2
git clone https://gitee.com/hf-models/roberta-large-mnli.git local_models/roberta-large-mnli
```

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入 DeepSeek API Key
```

### 运行

```bash
python gui.py                  # GUI 界面
python main.py --mode test     # 命令行：测试模式
python main.py --mode user --text "你的对话..."  # 用户模式
```

## 评分维度

| 维度 | 权重 | 测量什么 |
|------|------|---------|
| Retention（信息保留率） | 50% | 嵌入向量余弦相似度：记忆覆盖了多少对话信息 |
| Entity F1（实体覆盖） | 25% | spaCy NER：人名/数字/日期是否丢失 |
| Consistency（一致性） | 25% | RoBERTa NLI：记忆是否编造了不存在的事实 |

## 项目结构

```
├── main.py              # 命令行入口
├── gui.py               # GUI 入口
├── config.py            # 配置
├── evaluator/           # 评分器 (AutoMemo: 嵌入+NLI+NER)
├── generator/           # LLM 客户端 + Prompt 模板
├── experience/          # 经验库 (ADD/DELETE/MODIFY/KEEP)
├── optimizer/           # 迭代优化器 + 裁判 LLM
├── visualizer/          # 图表绘制
└── data/                # 数据集加载 + 测试样本
```
