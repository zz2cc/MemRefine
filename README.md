# MemRefine

> Iterative memory optimization via self-comparison — no model retraining required.

Inspired by [3DrawAgent](https://arxiv.org/abs/2604.08042)'s Contrastive Knowledge Extraction (CKE):
Generate K candidates → Hybrid scorer ranks them → Judge compares high vs low → Extract writing rules → Inject into next round.

---

## Demo

| Score Trajectory | Library Growth |
|---|---|
| ![trajectory](assets/demo_trajectory.png) | ![library](assets/demo_library.png) |

- **Left**: Composite score (black) across rounds, Retention (blue) + Entity F1 (green) breakdown
- **Right**: Rules in experience library (purple) + weakness distribution (bars)

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Download models (~2GB from Gitee):

```bash
mkdir -p local_models
git clone https://gitee.com/hf-models/all-MiniLM-L6-v2.git local_models/all-MiniLM-L6-v2
git clone https://gitee.com/hf-models/roberta-large-mnli.git local_models/roberta-large-mnli
```

Configure API key:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com/v1
JUDGE_API_KEY=sk-your-judge-key-here
```

Run:

```bash
python server.py        # Web UI (recommended)
python main.py --mode test     # 5 built-in dialogues
python main.py --mode user --text "..."    # single input
```

## Usage

### Web UI (`start.bat` or `python server.py`)

- **Test mode**: optimize 5 built-in English dialogues
- **User mode**: paste text or drag-drop files (PDF/PPTX/DOCX/CSV/JSON/XML/TXT/MD)
- Real-time log streaming, auto-loads results on completion

### CLI

```bash
python main.py --mode test
python main.py --mode user --text "your text..."
python main.py --mode user --file document.pdf
python main.py --dry-run          # scorer only
python main.py --mock-llm         # offline test
```

Optional: `--rounds 5` `--candidates 5` `--model deepseek-chat`

## Scoring

| Dimension | Weight | Method | Measures |
|---|---|---|---|
| Retention | 70% | MiniLM embedding cosine similarity | How much source content is covered |
| Entity F1 | 30% | spaCy NER match | Are names/numbers/dates preserved |

Embedding similarity is the text analogue of CLIP scoring in 3DrawAgent — fast (0.05s), interpretable.
Entity F1 catches what embeddings miss: "order #88421" → "the order" loses critical detail.

## Core Mechanism

Each round:

1. **Generate** K candidates (temperature anneals 0.8 → 0.3)
2. **Score** with Hybrid scorer (70% retention + 30% entity F1)
3. **Judge** compares top vs bottom candidates, extracts ≤5 abstract writing rules
4. **Detect weakness** (Retention or Entity F1 lower?), target next round
5. **Update library**: ADD (≤3/round) + KEEP (survival count) + DELETE (negative score) + MODIFY (merge near-duplicates)

**Temperature annealing**: T starts at 0.8 (explore diversity), drops to 0.3 (exploit precision). Paper reference: 3DrawAgent uses T=0.7 exploration / T=0.3 inference.

## Output

```
output/
├── test/                             # Test mode (5 dialogues)
│   ├── results.json
│   ├── results_experiences.json
│   ├── score_trajectory.png
│   ├── component_breakdown.png
│   ├── experience_growth.png
│   └── weakness_distribution.png
│
└── user/                             # User mode (single)
    ├── results.json
    ├── trajectory.png
    └── library.png
```

## Project Structure

```
├── server.py            # Web UI (Flask + SSE)
├── main.py              # CLI entry
├── config.py            # Configuration
├── start.bat            # Windows one-click launcher
├── evaluator/
│   ├── hybrid_scorer.py # Hybrid scorer (embedding + NLI)
│   └── ner.py           # Entity extraction (spaCy)
├── generator/
│   ├── llm_client.py    # LLM API client (parallel requests)
│   ├── prompts.py       # Prompt templates
│   └── mock_llm.py      # Mock LLM for offline testing
├── experience/
│   └── library.py       # Experience library (ADD/DELETE/MODIFY/KEEP)
├── optimizer/
│   ├── iterator.py      # Core iteration loop
│   └── judge.py         # Judge LLM
├── visualizer/
│   └── plotting.py      # matplotlib charts
├── utils.py             # Utilities (file parser, seed, logging)
├── data/                # Dataset loader + test samples (EN)
├── local_models/        # Downloaded models
├── output/              # Results
└── assets/              # README screenshots
```

## Limitations

1. **English only**: scoring components (MiniLM/spaCy/RoBERTa) are English models. Chinese input must be translated first.
2. **Slow due to API latency**: ~40-60s per round, 6 LLM calls each. Bottleneck is API wait time.
3. **Modest quality gains**: positive trends observed but randomness cannot be excluded. Scorer-generator alignment may be the core bottleneck.

## FAQ

**Rules=0?** Check `.env` API key. Invalid key → mock LLM fallback.

**Supported file formats?** PDF, PPTX, DOCX, CSV, JSON, XML, TXT, MD. Drag-drop in Web UI.

**Use for lecture notes?** Yes, `--mode user` with PPT text. See archived `study-notes` branch.

## Reference

- [3DrawAgent](https://arxiv.org/abs/2604.08042) — Contrastive Knowledge Extraction + experience library iteration

---

# MemRefine · 中文说明

> 迭代式记忆精炼——无需重新训练模型，通过多轮自我对比迭代优化文本摘要质量。

借鉴 [3DrawAgent](https://arxiv.org/abs/2604.08042) 的对比知识提取范式：生成 K 个候选 → 混合评分排序 → 裁判对比高低分 → 提取通用写作规则 → 注入下一轮 prompt。

## 效果演示

| 得分轨迹 | 经验库增长 |
|---|---|
| ![trajectory](assets/demo_trajectory.png) | ![library](assets/demo_library.png) |

- **左图**：Composite 得分（黑线）随轮次变化，Retention 信息保留率（蓝）+ Entity F1 实体覆盖（绿）
- **右图**：经验库规则数量（紫线）+ 短板分布（柱状图）

## 快速开始

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

从 Gitee 镜像下载模型（约 2GB）：

```bash
mkdir -p local_models
git clone https://gitee.com/hf-models/all-MiniLM-L6-v2.git local_models/all-MiniLM-L6-v2
git clone https://gitee.com/hf-models/roberta-large-mnli.git local_models/roberta-large-mnli
```

配置 API Key：

```bash
cp .env.example .env
# 编辑 .env 填入 DeepSeek API Key
```

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com/v1
JUDGE_API_KEY=sk-your-judge-key-here
```

运行：

```bash
python server.py        # Web 界面（推荐）
python main.py --mode test     # 5 条内置对话
python main.py --mode user --text "..."    # 单条输入
```

## 使用方式

### Web 界面（双击 `start.bat` 或 `python server.py`）

- **Test 模式**：对 5 条内置英文对话运行优化
- **User 模式**：粘贴文本或拖拽上传文件（PDF/PPTX/DOCX/CSV/JSON/XML/TXT/MD）
- 实时日志流式展示，完成后自动加载结果面板

### 命令行

```bash
python main.py --mode test
python main.py --mode user --text "文本内容..."
python main.py --mode user --file document.pdf
python main.py --dry-run          # 仅测试评分器
python main.py --mock-llm         # 离线快速验证
```

可选参数：`--rounds 5` `--candidates 5` `--model deepseek-chat`

## 评分机制

| 维度 | 权重 | 方法 | 测量内容 |
|---|---|---|---|
| Retention 信息保留率 | 70% | MiniLM 嵌入余弦相似度 | 记忆覆盖了原文中多少信息 |
| Entity F1 实体覆盖 | 30% | spaCy NER 实体匹配 | 人名、数字、日期等关键事实是否丢失 |

嵌入相似度是 3DrawAgent 中 CLIP 评分的文本类比——快（0.05 秒/次）、可解释。实体 F1 捕获嵌入遗漏的细节："订单 #88421" 变成 "订单" 时嵌入察觉不到但实体匹配能暴露差距。

## 核心机制

每轮五个阶段：

1. **生成** K 个候选（温度退火 0.8 → 0.3，探索到精准）
2. **评分** Hybrid 混合评分器（70% 信息保留 + 30% 实体覆盖）
3. **裁判** 对比高低分候选，提取 ≤5 条抽象写作规则
4. **检测短板**（Retention / Entity F1 哪个更低），下一轮针对性注入
5. **更新经验库**：ADD（每轮 ≤3 条）+ KEEP（存活计数）+ DELETE（淘汰负分）+ MODIFY（合并相似规则）

**温度退火**：Round 0 用 T=0.8 探索多样性，逐轮降至 Round 4 用 T=0.3 精准生成（3DrawAgent 论文参考：探索 T=0.7，推理 T=0.3）。

## 输出结构

```
output/
├── test/                             # 测试模式（5 条对话）
│   ├── results.json                  # 完整得分 + 最佳记忆
│   ├── results_experiences.json      # 每条对话的规则
│   ├── score_trajectory.png          # 五色线得分轨迹
│   ├── component_breakdown.png       # 六面板成分分解
│   ├── experience_growth.png         # 经验库增长
│   └── weakness_distribution.png     # 短板频率
│
└── user/                             # 用户模式（单条）
    ├── results.json
    ├── trajectory.png                # 得分 + 成分分解
    └── library.png                   # 规则增长 + 短板
```

## 项目结构

```
├── server.py            # Web 界面（Flask + SSE）
├── main.py              # 命令行入口
├── config.py            # 配置
├── start.bat            # Windows 一键启动
├── evaluator/
│   ├── hybrid_scorer.py # 混合评分器（嵌入 + NLI）
│   └── ner.py           # 实体抽取（spaCy）
├── generator/
│   ├── llm_client.py    # LLM API 客户端（并行请求）
│   ├── prompts.py       # Prompt 模板
│   └── mock_llm.py      # 假 LLM（离线测试用）
├── experience/
│   └── library.py       # 经验库（ADD/DELETE/MODIFY/KEEP）
├── optimizer/
│   ├── iterator.py      # 核心迭代循环
│   └── judge.py         # 裁判 LLM
├── visualizer/
│   └── plotting.py      # matplotlib 图表
├── utils.py             # 工具函数（文件解析/种子/日志）
├── data/                # 数据集 + 测试样本（英文）
├── local_models/        # 需自行下载的模型
├── output/              # 输出结果
└── assets/              # README 截图
```

## 局限与待改进

1. **仅支持英文**：评分组件（MiniLM/spaCy/RoBERTa）均为英文模型，中文输入需先翻译。替换为多语言模型（`paraphrase-multilingual-MiniLM`、`zh_core_web_sm` 等）可解决
2. **速度受 API 限制**：每轮 6 次 LLM 调用，约 40-60 秒/轮，瓶颈全在等待 API 响应
3. **质量提升有限**：迭代有正向趋势但不排除随机性，部分对话触发早停。评分器与生成目标的对齐度是核心瓶颈

## 常见问题

**Rules=0？** 检查 `.env` 中 API Key 是否正确，Key 无效时自动回退到 mock LLM（无真实规则）。

**支持哪些文件格式？** PDF、PPTX、DOCX、CSV、JSON、XML、TXT、MD，Web 界面拖拽上传自动提取文字。

**能做上课笔记吗？** 可以，用 `--mode user` 输入 PPT/讲课文本即可。已归档的 `study-notes` 分支有专用模式。

## 参考

- [3DrawAgent](https://arxiv.org/abs/2604.08042) — 对比知识提取 + 经验库迭代
