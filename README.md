# Synaxis-Syllabus2017

`1.0.0-hotfix1`（Python 包版本 `1.0.0.post1`）。将指定的 2017 年版普通高中课程标准中必修、选择性必修及适用的共用课程内容转换为可追溯的本地索引，支持中文词项、语义与混合检索。

自编程序和提取规则采用 **AGPL-3.0-or-later**，详见 [LICENSE](LICENSE)；修改版本及网络服务须遵守其相应源代码义务。

这是课程标准处理工具，不是独立高考考纲、地方考试细则或官方认定服务。检索未命中不代表知识点不考；分数仅用于排序。

仓库只提供程序、定向提取规则、来源文件名/哈希和合成测试。PDF、课标全文及其 JSONL/SQLite/向量派生物、ONNX 模型、个人环境和运行报告均不在公开源码内。课程资料和模型由部署者按各自授权取得，不能将本项目的代码许可套用于这些资源。完整授权边界见 [NOTICE](NOTICE) 和 [LICENSE](LICENSE)。

## 从源码安装

需要 Python 3.12.15 及 [uv 0.13.5+](https://docs.astral.sh/uv/getting-started/installation/)。从仓库根目录运行：

```sh
git clone https://github.com/STMfan/Synaxis-Syllabus2017.git
cd Synaxis-Syllabus2017
uv venv .venv --python 3.12.15
uv sync --frozen
uv run --frozen curriculum-search --version
uv run --frozen python -m unittest discover -s tests -v
```

基础安装仅包含词项检索所需的 NumPy；默认测试只用合成内容、临时数据库和模拟向量，不需要课标、模型或付费接口。PDF 提取和本地向量生成按需安装：

```sh
uv sync --frozen --extra extract --extra embedding
```

安全基线：Pillow 13+、Certifi 2026.10.19+，具体版本由 `uv.lock` 固定。FastEmbed 0.9.0 的旧 Pillow 上限暂由 `tool.uv.override-dependencies` 覆盖，本项目只使用文本嵌入，并验证实际本地 ONNX 检索；待上游支持后移除覆盖。CI 审计全部可选依赖。

`extract` 使用 PyMuPDF，`embedding` 使用 FastEmbed/ONNX Runtime。可选安装不会改变相关组件的许可义务；公开分发、修改或部署时须按 LICENSE/NOTICE 评估整个组合的授权要求。

## 在本地重建课程索引

1. 自行取得有权使用的原始课标 PDF，存入例如 `sources/`。先查看本项目审查过的文件名、页数和 SHA-256：

   ```sh
   uv run --frozen curriculum-search sources
   ```

   提取规则与目录页、页码边界、英语表格布局绑定；程序会在读取 PDF 页前拒绝哈希或文件名不符的版本。官网重新生成的 PDF 即使标题相同，字节也可能不同；不能靠重命名绕过核验，应先审查规则与来源。原始 PDF 不会被修改。

2. 定向提取与建立 SQLite FTS5 词项索引：

   ```sh
   uv run --frozen --extra extract curriculum-search extract --input sources --output data
   uv run --frozen curriculum-search index --data data
   uv run --frozen curriculum-search search "遗传信息" --data data --subject 生物学 --category 必修课程 --mode lexical --top-k 3
   ```

   `data/` 必须是新目录或空目录；升级时生成新的数据目录，不覆盖正在使用的索引。`extract` 只读取规则指定的目录、边界和目标页，清单记录每次实际读取的 PDF 页及原因。未知文件和版式不符会报错。

3. 如需语义或混合检索，首次明确允许下载模型并生成向量：

   ```sh
   uv run --frozen --extra embedding curriculum-search embed --data data --cache models --allow-download
   uv run --frozen --extra embedding curriculum-search search "生命怎样保存和传递遗传信息" --data data --model-cache models --mode hybrid --top-k 3
   ```

   默认模型为 `BAAI/bge-small-zh-v1.5`，FastEmbed 使用 Qdrant ONNX 分发的固定修订 `46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59`，512 维、L2 归一化；文档使用 `passage_embed`，查询使用 `query_embed`。首次安装依赖和下载模型需要网络，此后不加 `--allow-download`，模型仅从本地加载。查询不会使用云端 embedding API，也不会自动下载缺少的模型。

   语义检索超过 512 token 会明确拒绝，不静默截断。缺少向量或模型时，语义/混合模式报错；可以明确选择 `--mode lexical`。更换模型必须重新建立向量；Synaxis 后端目前只接受上述 512 维模型格式。

可设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 进一步禁止模型库联网。`lexical` 模式完全不加载模型。

## 数据格式与引用

| 本地文件 | 用途 |
|---|---|
| `data/manifest.json` | 来源文件哈希、版本、页码边界、实际读取记录与局限 |
| `data/sections.jsonl` | 父章节、页级片段、分类依据 |
| `data/chunks.jsonl` | 检索片段、类别、章节路径及来源 |
| `data/source_pages.jsonl` | 定向提取页的原始文本及正文坐标，供本地核对 |
| `data/markdown/` | 本地生成的可读章节内容 |
| `data/curriculum.sqlite3` | 词项索引 |
| `data/vectors.npy`、`vectors.json` | 向量、ID 顺序、编码方式与文件哈希 |
| `models/` | 本地模型缓存 |

返回结果保留 `source.file`、`source.sha256`、从 1 开始的 `source.pdf_pages`，以及独立的印刷页码 `printed_pages`。引用时保留这些字段、`heading_path` 和 `quality_warnings`。PDF 页序不等于印刷页码。

数学公式和特殊字形可能存在原 PDF 编码损伤，程序保留警告；它不凭猜测补字、不做 OCR、不生成替代图示。精确公式、表格或适用范围应回到原页核对。原文与分块重叠上下文分开保存；课程共享范围不重复复制成多份。

## 接入 Synaxis 后端

在完成索引和向量构建后，导出一个新的私有便携目录：

```sh
uv run --frozen curriculum-search bundle --data data --model-cache models --output dist/syllabus2017-local
```

输出根目录直接包含 `data/`、扁平的 `models/` 和 `import.json`。导出器核验数据、向量、ID、来源和数据库内容，将模型缓存链接复制为普通文件；不复制 PDF、虚拟环境、下载缓存或历史验收报告，不覆盖已有目标。收据逐文件记录 SHA-256 和模型 revision；`models/MODEL-LICENSE.txt` 保留模型上游版权与完整 MIT 条款，后端导入时继续携带并校验。该本地数据包是部署材料，不能当作本仓库的开源附件公开发布。

在独立的 [Synaxis-backend](https://github.com/STMfan/Synaxis-backend) 仓库根目录运行：

```sh
uv run --frozen python -m runtime.import_syllabus /absolute/path/to/syllabus2017-local --destination /absolute/path/to/private/syllabus2017
```

然后在后端私有 `config/runtime/syllabus.json` 设置：

```json
{
  "syllabus_enabled": true,
  "syllabus_data_dir": "/absolute/path/to/private/syllabus2017/data",
  "syllabus_model_cache": "/absolute/path/to/private/syllabus2017/models",
  "syllabus_mode": "hybrid",
  "syllabus_top_k": 5,
  "syllabus_threads": 2
}
```

停止后端后切换目录并重启。无需运行本项目 HTTP 服务；后端只读本地数据和模型，不直接导入本项目的程序。原有 HF 缓存格式的历史包仍可通过后端兼容导入命令迁移。`/health` 和带学生认证的 `/v1/capabilities` 可核对实际就绪状态。

## 供其它项目调用

```python
from curriculum_search import CurriculumSearch

search = CurriculumSearch(data_dir="/absolute/path/to/data", model_cache="/absolute/path/to/models")
hits = search.search("圆锥曲线", subject="数学", course_category="选择性必修课程", mode="hybrid", top_k=3)
section = search.get_section(hits[0]["section_id"]) if hits else None
```

也可运行仅监听本机的轻量 HTTP 接口：

```sh
uv run --frozen --extra embedding curriculum-search serve --data data --model-cache models --host 127.0.0.1 --port 8765
```

`GET /health`、`POST /search`、`GET /chunks/{id}`、`GET /sections/{id}` 可供本机项目调用。`/search` 请求字段为 `query`、`subject`、`course_category`、`mode` 和 `top_k`。参数错误返回 400，条目不存在返回 404，索引/模型缺失返回 503。服务不提供用户认证或公网访问控制，使用单请求线程避免共享 tokenizer 并发修改；不要直接将其暴露到公网。

## 维护与测试

源码根目录直接包含 `pyproject.toml`、`uv.lock`、`src/`、`tests/` 和 CI；不保留过程报告、重复部署文档或已生成的数据目录。修改接口或索引格式时运行：

```sh
uv run --frozen python -m unittest discover -s tests -v
```

测试覆盖合成词项与向量检索、分类过滤、来源保留、索引过期与损坏拒绝、HTTP 参数/错误、离线模型边界和可移植 bundle 收据。它们不证明真实 PDF 已逐字人工校对，也不证明模型结果符合教学标准。
