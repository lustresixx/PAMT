[English](README.md) | **中文**
# PAMT 偏好感知记忆树
项目名：**PAMT（偏好感知记忆树）**。Python 包名为 `pamt`。

一个可运行的 **偏好感知记忆更新系统**，集成了记忆树、偏好控制 Prompt 和可视化 UI，同时提供可嵌入任何 agent 的插件接口。

---

## 亮点

- 偏好提取（模型 + 回退）
- SW/EMA 融合 + 变化检测
- 三层记忆树（root -> category -> leaf）
- 叶子合并与标签粗化
- 可视化聊天界面
- 外部插件接口（memory plugin）
- JSON 持久化（自动读写）

---

## 目录

- [快速开始](#快速开始)
- [可视化聊天](#可视化聊天)
- [外部插件用法](#外部插件用法)
- [系统设计](#系统设计)
- [请求流程](#请求流程)
- [配置项](#配置项)
- [项目结构](#项目结构)
- [说明](#说明)

---

## 快速开始

### CLI 示例（带持久化）

```bash
python example/chatbot.py \
  --llm-provider ollama --llm-model qwen2.5:3b \
  --embed-provider ollama --embed-model nomic-embed-text \
  --memory-file data/pamt_memory.json
```

---

## 可视化聊天

```bash
python example/visual_chatbot.py \
  --llm-provider ollama --llm-model qwen2.5:3b \
  --embed-provider ollama --embed-model nomic-embed-text
```

浏览器打开：
```text
http://127.0.0.1:7860
```

界面功能：
- Memory Tree / Baseline 切换
- 是否带上下文可切换
- Node Inspector 查看 Fusion / SW / EMA
- Debug 面板查看路由与更新
![img.png](asset/img.png)
---

## 外部插件用法

插件提供两个核心能力：
- `augment(query)` -> 返回“带记忆的 prompt”
- `update(query, response)` -> 写回记忆

```python
from pamt import create_memory_plugin, ModelConfig, EmbeddingConfig

plugin = create_memory_plugin(
    model_config=ModelConfig(provider="ollama", model_name="qwen2.5:3b"),
    embedding_config=EmbeddingConfig(provider="ollama", model_name="nomic-embed-text"),
    storage_path="data/pamt_memory.json",
)

aug = plugin.augment("帮我写封邮件")
prompt = aug.prompt

# 交给任何 agent / LLM
response = "..."
plugin.update("帮我写封邮件", response)
```

---

## 系统设计

### 1) 偏好提取
维度：语气、情绪、长度、密度、正式度。

模型路径：
- 优先模型：RoBERTa 语气 + SKEP 情绪 + OpenNRE 密度 + 正式度模型
- 回退：HF/GLiNER -> 启发式

长度优先使用模型 token 数，否则用词数。

### 2) 偏好更新（SW + EMA + 变化检测）

连续维度：
- `SW_t(d) = mean(p_{t-W+1:t}(d))`
- `EMA_t(d) = alpha * EMA_{t-1}(d) + (1-alpha) * p_t(d)`
- `w_t(d) = lambda * SW_t(d) + (1-lambda) * EMA_t(d)`

类别维度：对概率向量做 SW/EMA。

变化检测：
- `Delta_t(d) = |SW_t(d) - EMA_t(d)|`
- `C_t(d) = Delta_t(d) / (epsilon + Var(SW) + Var(EMA))`
- `C_t(d) > threshold` 触发变化

### 3) 记忆树

结构：
- root -> category -> leaf

路由：
- Strict 匹配 -> 下沉
- Loose 匹配 -> 合并候选
- 否则 fallback

叶子管理：
- 标签由 **query + response** 推断
- 叶子用 **内容 embedding** 复用
- 超过最大叶子数时合并
- 相似度低时用 LLM 生成更粗标签

更新权重：
- Leaf：1.0
- Category：`1 / category_leaf_count`
- Root：`1 / total_leaf_count`

### 4) Prompt 注入
若已有偏好，将控制描述插入用户输入前；否则直接使用原始输入。

### 5) 持久化
树结构 + 偏好状态保存到 JSON。插件自动 load-or-create、update 后自动保存。

---

## 请求流程

1) 用户 query 进入
2) 记忆检索（短 query 且 fallback 时用上下文）
3) 生成带偏好的 prompt
4) LLM 生成回复
5) 抽取偏好信号
6) 记忆树路由 + 更新
7) 保存到磁盘

---

## 配置项

- `pamt/config.py`：窗口大小、EMA 衰减、变化阈值、Prompt 分桶
- `RetrievalConfig`：strict/loose 相似度、最大叶子数、合并阈值

---

## 项目结构

```
pamt/
  core/                 # prompting, memory tree, update logic
  extractors/           # preference extraction
  embeddings/           # embedding clients
  llms/                 # LLM clients
  memory_plugin.py      # external plugin wrapper
example/
  chatbot.py            # CLI 示例
  visual_chatbot.py     # 可视化 UI 服务
prompts/
  category_leaf.txt     # 标签推断
  leaf_only.txt
  leaf_merge_summary.txt
```

---

## 说明

这是一个偏好记忆更新机制的 **可运行复现 + 工程化扩展**。


