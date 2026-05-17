# Iterative Memory Optimization via Self-Comparison

通过多轮自我对比迭代，在不重新训练 LLM 的前提下提升对话记忆文档的生成质量。

借鉴 3DrawAgent 的范式：**生成候选 → 评分排序 → 裁判对比高低分 → 提取写作规则 → 注入下一轮 prompt**。

## 环境要求

- Python **3.12**（建议，其他版本需自行适配）
- Git LFS（用于下载模型大文件）

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 下载模型（从 Gitee 镜像，约 2GB）

```bash
mkdir -p local_models
git clone https://gitee.com/hf-models/all-MiniLM-L6-v2.git local_models/all-MiniLM-L6-v2
git clone https://gitee.com/hf-models/roberta-large-mnli.git local_models/roberta-large-mnli
```

## 配置 API Key

本项目使用 DeepSeek API（OpenAI 兼容接口）。支持生成 LLM 和裁判 LLM 使用不同 key。

```bash
cp .env.example .env
```

编辑 `.env`：

```
OPENAI_API_KEY=sk-your-deepseek-key-here
OPENAI_BASE_URL=https://api.deepseek.com/v1
JUDGE_API_KEY=sk-your-judge-key-here
```

## 运行

### GUI（推荐）

```bash
python server.py
```

- Test 模式：对 5 条内置英文对话运行迭代优化
- User 模式：输入单条对话文本或从文件加载
- 实时显示进度日志，完成后自动切换到结果标签页
- 结果标签页：最优记忆文本 / 输出文件（双击打开）/ 得分汇总 / 学到的规则

### 命令行

```bash
python main.py --mode test                              # 测试模式（5 条内置对话）
python main.py --mode user --text "你的对话文本..."       # 用户模式：直接输入
python main.py --mode user --file my_dialogue.txt       # 用户模式：从文件读取
python main.py --dry-run                                # 只测试评分器，不调 LLM
python main.py --mock-llm                               # 用假 LLM 快速验证流程
```

可选参数：`--rounds 5`（迭代轮次）`--candidates 5`（每轮候选数）

## 输出

```
output/
├── test/                             # 测试模式输出
│   ├── results.json                  # 5 条对话的完整得分数据
│   ├── results_experiences.json      # 每条对话独立学到的规则集
│   ├── score_trajectory.png          # 5 色线各自得分轨迹
│   ├── component_breakdown.png       # 评分三维度分解
│   ├── experience_growth.png         # 经验库规则数量增长
│   └── weakness_distribution.png     # 各维度短板频率
│
└── user/                             # 用户模式输出
    ├── results.json                  # 单条对话完整数据
    ├── trajectory.png                # 得分轨迹 + 三维度分解
    └── library.png                   # 规则增长 + 短板分布
```

## 评分机制

| 维度 | 权重 | 方法 | 测什么 |
|------|------|------|--------|
| Retention | 70% | MiniLM 嵌入余弦相似度 | 记忆覆盖了对话中多少信息 |
| Entity F1 | 30% | spaCy NER 实体匹配 | 人名/数字/日期是否丢失 |

## 当前局限

- 评分器组件（MiniLM / spaCy / RoBERTa）均为英文模型，**直接输入中文评分会不准**
- 建议将中文对话翻译为英文后输入，或替换为对应的多语言模型

## 项目结构

```
├── main.py              # 命令行入口
├── server.py            # Web 界面入口
├── config.py            # 配置
├── _ssl_patch.py        # SSL 证书修复（受限网络环境用）
├── evaluator/
│   ├── automemo.py      # AutoMemo 评分器（嵌入 + NLI）
│   └── ner.py           # 实体抽取（spaCy）
├── generator/
│   ├── llm_client.py    # LLM API 客户端（OpenAI 兼容）
│   ├── prompts.py       # Prompt 模板
│   └── mock_llm.py      # 假 LLM（离线测试用）
├── experience/
│   ├── library.py       # 经验库（ADD/DELETE/MODIFY/KEEP）
│   └── operations.py    # 经验管理器
├── optimizer/
│   ├── iterator.py      # 核心迭代循环
│   └── judge.py         # 裁判 LLM
├── visualizer/
│   └── plotting.py      # matplotlib 图表
├── data/
│   ├── dialog_data.py   # 数据集加载器
│   └── samples/         # 测试样本
├── local_models/        # 本地模型（需自行下载）
└── output/              # 输出结果
```
