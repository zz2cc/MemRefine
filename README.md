# MemRefine · 迭代式记忆精炼

> Iterative memory optimization via self-comparison — no model retraining required.  
> 无需重新训练模型，通过多轮自我对比迭代优化文本摘要质量。

Inspired by [3DrawAgent](https://arxiv.org/abs/2604.08042)'s Contrastive Knowledge Extraction (CKE) paradigm:  
借鉴 3DrawAgent 的对比知识提取范式：

```
Generate K candidates → Hybrid Scorer ranks them → Judge compares high vs low → Extract writing rules → Inject into next round
生成 K 个候选 → 混合评分排序 → 裁判对比高低分 → 提取通用写作规则 → 注入下一轮 prompt
```

## Demo · 效果演示

| Score Trajectory · 得分轨迹 | Library Growth · 经验库增长 |
|---|---|
| ![trajectory](assets/demo_trajectory.png) | ![library](assets/demo_library.png) |

- **Left**: Composite score (black) across rounds, Retention (blue) + Entity F1 (green) breakdown  
  **左图**：Composite 得分（黑线）随轮次变化，Retention 信息保留率（蓝）+ Entity F1 实体覆盖（绿）
- **Right**: Rules in experience library (purple) + weakness distribution (bars)  
  **右图**：经验库规则数量（紫线）+ 短板分布（柱状图）

## Quick Start · 快速开始

### 1. Environment · 环境

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Models · 下载模型 (~2GB from Gitee mirror · 从 Gitee 镜像)

```bash
mkdir -p local_models
git clone https://gitee.com/hf-models/all-MiniLM-L6-v2.git local_models/all-MiniLM-L6-v2
git clone https://gitee.com/hf-models/roberta-large-mnli.git local_models/roberta-large-mnli
```

### 3. API Key · 配置密钥

```bash
cp .env.example .env
# Edit .env with your DeepSeek API key · 填入 DeepSeek API Key
```

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com/v1
JUDGE_API_KEY=sk-your-judge-key-here  # optional · 可选，裁判用不同 Key
```

### 4. Run · 运行

```bash
python server.py        # Web UI (recommended) · Web 界面（推荐）
# or · 或
python main.py --mode test     # 5 built-in dialogues · 5 条内置对话
python main.py --mode user --text "..."    # single input · 单条输入
```

## Usage · 使用方式

### Web UI · Web 界面 (double-click `start.bat` or `python server.py` · 双击 `start.bat`)

- **Test mode**：optimize 5 built-in English dialogues · 对 5 条内置对话运行优化
- **User mode**：paste text or drag-drop files (PDF/PPTX/DOCX/CSV/JSON/XML/TXT/MD) · 粘贴或拖拽上传
- Real-time log streaming, auto-loads results on completion · 实时日志，完成后自动展示结果

### CLI · 命令行

```bash
python main.py --mode test                         # test mode · 测试模式
python main.py --mode user --text "your text..."    # direct input · 直接输入
python main.py --mode user --file document.pdf     # from file · 从文件
python main.py --dry-run                           # scorer only · 仅测试评分器
python main.py --mock-llm                          # offline · 离线快速验证
```

Optional: `--rounds 5` `--candidates 5` `--model deepseek-chat`

## Scoring · 评分机制

| Dimension · 维度 | Weight · 权重 | Method · 方法 | Measures · 测什么 |
|---|---|---|---|
| Retention · 信息保留率 | 70% | MiniLM embedding cosine similarity · 嵌入余弦相似度 | How much of the source is covered · 原文信息覆盖度 |
| Entity F1 · 实体覆盖 | 30% | spaCy NER match · 实体匹配 | Are names/numbers/dates preserved · 关键事实是否丢失 |

> **Why this design?** · 为什么这样设计  
> Embedding similarity is the text analogue of CLIP scoring in 3DrawAgent — fast (0.05s), interpretable.  
> 嵌入相似度是 3DrawAgent 中 CLIP 评分的文本类比——快（0.05 秒）、可解释。  
> Entity F1 catches what embeddings miss: "order #88421" → "the order".  
> 实体 F1 捕获嵌入遗漏的：原文的"订单 #88421"在记忆中变成了"订单"。

## Core Mechanism · 核心机制

### Iteration Loop · 迭代循环

Each round · 每轮：

```
Phase 1 — Generate K candidates (temperature anneals 0.8 → 0.3)
          生成 K 个候选（温度退火：探索 0.8 → 精准 0.3）
Phase 2 — Hybrid scorer ranks them · 混合评分排序
Phase 3 — Judge LLM compares top vs bottom, extracts ≤5 abstract writing rules
          裁判 LLM 对比高低分，提取 ≤5 条通用写作规则
Phase 4 — Detect weakness (Retention or Entity F1 lower?), target next round
          检测短板（Retention/Entity F1 哪个更低），针对性注入
Phase 5 — ADD (≤3/round) + KEEP (survival count) + DELETE (negative score) + MODIFY (merge similar)
          ADD（每轮≤3条）+ KEEP（存活计数）+ DELETE（淘汰负分）+ MODIFY（合并相似）
```

### Experience Library · 经验库四操作

| Operation · 操作 | When · 触发时机 | What it does · 作用 |
|---|---|---|
| ADD | After judge · 裁判产出后 | ≤3 rules added, semantic dedup · 最多 3 条入库，语义去重 |
| KEEP | End of round · 每轮结束 | usage_count +1 for all rules, harder to delete · 计数 +1，活得越久越难被删 |
| DELETE | End of round · 每轮结束 | Remove rules with negative score_delta after ≥2 rounds · 存活 ≥2 轮且负分的规则 |
| MODIFY | Library full · 库满 10 条 | Merge near-duplicates (cos > 0.75), keep higher-scored · 合并相似规则，保留高分 |

### Temperature Annealing · 温度退火

```
Round 0: T=0.8 (explore diversity · 探索多样性)
Round 1: T=0.675
Round 2: T=0.55
Round 3: T=0.425
Round 4: T=0.3  (exploit precision · 精准生成)
```

> Paper reference · 论文依据: T=0.7 for exploration, T=0.3 for inference (3DrawAgent)

## Output · 输出结构

```
output/
├── test/                             # Test mode · 测试模式 (5 dialogues)
│   ├── results.json                  # Full scores + best memories · 完整得分 + 最佳记忆
│   ├── results_experiences.json      # Per-dialogue rules · 每条对话的规则
│   ├── score_trajectory.png          # 5 colored lines per dialogue · 五色线得分轨迹
│   ├── component_breakdown.png       # 6 panels: retention/entity_f1 · 六面板成分分解
│   ├── experience_growth.png         # Library size per dialogue · 经验库增长
│   └── weakness_distribution.png     # Weakest dimension frequency · 短板频率
│
└── user/                             # User mode · 用户模式 (single)
    ├── results.json
    ├── trajectory.png                # Score + component breakdown · 得分 + 成分分解
    └── library.png                   # Growth + weakness · 规则增长 + 短板
```

## Project Structure · 项目结构

```
├── server.py            # Web UI (Flask + SSE) · Web 界面
├── main.py              # CLI entry · 命令行入口
├── config.py            # Configuration · 配置
├── start.bat            # Windows one-click launcher · Windows 一键启动
├── _ssl_patch.py        # SSL fix for restricted networks · 受限网络 SSL 修复
├── evaluator/
│   ├── hybrid_scorer.py # Hybrid scorer (embedding + NLI) · 混合评分器
│   └── ner.py           # Entity extraction (spaCy) · 实体抽取
├── generator/
│   ├── llm_client.py    # LLM API client (parallel requests) · API 客户端
│   ├── prompts.py       # Prompt templates · Prompt 模板
│   └── mock_llm.py      # Mock LLM for offline testing · 离线测试用假 LLM
├── experience/
│   └── library.py       # Experience library (ADD/DELETE/MODIFY/KEEP) · 经验库
├── optimizer/
│   ├── iterator.py      # Core iteration loop · 核心迭代循环
│   └── judge.py         # Judge LLM (compare + parse) · 裁判 LLM
├── visualizer/
│   └── plotting.py      # matplotlib charts · matplotlib 图表
├── utils.py             # Utilities (file parser, seed, logging) · 工具函数
├── data/
│   ├── dialog_data.py   # Dataset loader · 数据集加载
│   └── samples/         # Test samples (EN) · 测试样本（英文翻译版）
├── local_models/        # Downloaded models · 需自行下载
├── output/              # Results · 输出
└── assets/              # README screenshots · README 截图
```

## Limitations · 局限与待改进

1. **English only · 仅支持英文**  
   Scoring components (MiniLM/spaCy/RoBERTa) are English models. Chinese input must be translated first.  
   评分组件均为英文模型，中文输入需先翻译。

2. **Slow due to API latency · 速度受 API 限制**  
   Each round needs 6 LLM calls (5 generate + 1 judge), ~40-60s per round. Bottleneck is API wait time.  
   每轮 6 次 LLM 调用，约 40-60 秒/轮，瓶颈全在等待 API 响应。

3. **Modest quality gains · 质量提升有限**  
   Current experiments show positive trends but randomness cannot be excluded. Some dialogues early-stop.  
   迭代有正向趋势但不排除随机性，部分对话触发早停。评分器与生成目标的对齐度是核心瓶颈。

## FAQ · 常见问题

**Q: Why is Rules=0? · 为什么规则数为 0？**  
A: Check `.env` API key. Invalid key → falls back to mock LLM (no real rules).  
检查 `.env` 中 API Key 是否正确。Key 无效时会回退到 mock LLM。

**Q: What file formats are supported? · 支持哪些文件格式？**  
A: PDF, PPTX, DOCX, CSV, JSON, XML, TXT, MD. Drag-drop in Web UI.

**Q: Can I use this for lecture notes? · 能做上课笔记吗？**  
A: Yes. Use `--mode user` with PPT/lecture text. For a dedicated mode, see `study-notes` branch (archived).  
可以。用 `--mode user` 输入 PPT/讲课文本即可。

## Branches · 分支说明

| Branch · 分支 | Content · 内容 |
|---|---|
| `main` | Primary: dialogue memory optimization + Web UI · 主干 |
| `rule-evolution` | Rule evolution tracking + temperature annealing (merged) · 规则演化追踪（已合并） |

## Reference · 参考

- [3DrawAgent](https://arxiv.org/abs/2604.08042) — Contrastive Knowledge Extraction + experience library iteration · 对比知识提取 + 经验库迭代
