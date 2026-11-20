# HiBlastNet 投稿前仓库一致性核对报告

首次核查：2026-11-19（America/Denver；链接测试时 UTC 已为 2026-11-20）。链接与源码打包修复更新：2026-11-20（America/Denver）。

## 结论

文档对齐及遗漏源码的打包修复已完成，干净源码快照的原图评级与单叶预测验证通过。用户随后在另一台电脑测试并确认两个 OneDrive 地址可用；尚未独立核对网盘下载包哈希，也未完成全部 12 个 checkpoint 的论文数值复现。

- 双语 README 的题目已同步，引用状态仍为 in preparation；消融表引用改为 Table 7。
- 复现脚本保存的 40 个参考数字（4 方法 × 5 指标 × mean/SD）与提示词完全一致，未改任何参考数字或容差。
- 实际计算了权重 ZIP 内的 12 个 SHA-256，全部与包内清单及本地解压文件一致；提示词明确提供的 6 个论文哈希全部匹配。
- 两份 README 中数据集和权重的旧 OneDrive 地址已替换为当前共享地址。已在管理页核对“拥有链接的人员都可查看”，并打开对应 ZIP 的下载页面；用户在另一台电脑反馈这两个链接可以使用。其余四个下载入口保持原样。
- **缺失 `blastseg/data/` 的发布阻塞已修复：** 将忽略规则限定为 `/data/`，从原 main 原样恢复 4 个源码文件；核心导入、4 个命令入口及实际预测验证均通过，未改算法。
- 自动化检查曾返回 HTTP 403，浏览器下载控制也曾超时；保留这些记录作为测试环境的限制，结合用户跨电脑测试结果，不将它们认定为分享链接故障。

## 工作边界和交付位置

- 任务依据：`E:/Download/REPO_ALIGNMENT_PROMPT_20261120.md`，论文事实以该提示词为准，没有修改论文。用户后续明确授权两个 OneDrive 地址修复，以及 `.gitignore` 和 4 个遗漏源码文件的打包修复，因此追加这些改动。
- 原工作目录：`D:/Code/Writing/HiBlastNet`，原分支 `main`。
- 对齐分支：`paper-alignment-20261120`，基于本地提交 `f752635cb6a6e68548c261a1824f71f7440f9dcc`。
- 本分支工作目录：`C:/Users/13763/.codex/visualizations/2026/10/22/01a16176-9587-7431-b274-0db501ad7b05/alignment_20261120/HiBlastNet`。
- 使用独立 Git worktree；原 `main` 上两份 README 的未提交修改均予以保留。仅将此次获准的两个 OneDrive 地址同步到本分支，没有整体合并原 `main` 的未提交改动。
- 初始审查和修复阶段没有推送。用户后续已明确授权将这些修复推送至 GitHub `main`；本报告随此次更新提交。未创建 tag、Release，也不改变仓库可见性。
- 训练、推理、评估、分级算法，配置、模型、权重和许可证均未修改。未加入专家研究内容。

本报告包含待决定的异常。不要把“文档检查通过”理解为“完整训练或 12 个 checkpoint 的重新评估已通过”。

## 必做项结果

| 项目 | 状态 | 结果 |
| --- | --- | --- |
| 标题 | 已改 | 两版 README 首屏说明、论文介绍和 Citation 均用当前题目，Citation 仍为 in preparation。 |
| 表号 | 已改 | 两份 README 和复现脚本的消融表引用均改为 Table 7；脚本符号同步改为 `PAPER_TABLE7`。 |
| 复现参考值 | 一致，无需改数字 | 40 个数字逐项一致；参考表是三种子 mean ± sample SD，没有逐 seed 期望值文件。 |
| 容差说明 | 已补充 | 默认 0.05，仅比较三种子均值；百分比指标单位为百分点，FP/leaf 单位为组件/叶；SD 不参与判定。 |
| 权重哈希 | 一致，已补入双语 README | ZIP 12/12 与清单一致，12/12 与解压文件一致，论文明确提供的 6/6 一致。 |
| README 论文事实 | 核对一致或补清口径 | 下表详列；没有擅自改论文数字。 |
| 下载链接 | 两个 OneDrive 地址已修复，用户跨电脑测试可用 | 双语 README 已同步当前地址；用户确认另一台电脑可用。下载包哈希和百度入口仍待核对。 |
| 源码打包遗漏 | 已修复，实际预测通过 | `/data/` 仅忽略根目录数据；恢复 4 个源码，干净快照导入/命令入口及原图评级、单叶预测均通过。 |

### 当前题目与横幅

HiBlastNet: a scale-preserving segmentation network for small rice blast lesions in quantitative measurement and severity grading

两份 README 的首屏 `strong` 说明、正文题目和 Citation 已同步。PNG 横幅本身只写 HiBlastNet、简短任务副标题和上方领域说明，**不含旧论文长题目**，因此未重新生成或修改任何图片。许可证段落逐字保持不变。

### 复现数值和容差

| 方法 ID | Dice (%) | Small survival (%) | Matched recall (%) | FP/leaf | Boundary F1 (%) |
| --- | --- | --- | --- | --- | --- |
| `gcnet` | 80.86 ± 0.10 | 55.18 ± 0.92 | 83.32 ± 0.17 | 10.80 ± 0.11 | 71.16 ± 0.53 |
| `gcnet_bc` | 81.68 ± 0.09 | 60.30 ± 0.33 | 85.39 ± 0.06 | 13.01 ± 0.15 | 72.46 ± 0.40 |
| `gcnet_lcjs` | 81.08 ± 0.04 | 57.93 ± 2.14 | 84.33 ± 0.36 | 12.26 ± 0.68 | 71.85 ± 0.73 |
| `hiblastnet` | 81.93 ± 0.17 | 65.43 ± 2.65 | 87.22 ± 0.18 | 15.27 ± 0.60 | 72.66 ± 0.14 |

- 数据来源：`scripts/check_reproduction.py:26` 的内嵌参考表，未引用外部参考数值文件；它与本次提示词的 Table 7 完全一致。
- 脚本遍历四种方法 × seeds 100/101/102，即 12 个 checkpoint；计算三次运行的均值和 n−1 样本标准差。
- Dice、Small survival、Matched recall、Boundary F1 在读取结果时乘 100，因此默认 `--tolerance 0.05` 为 **0.05 个百分点**。FP/leaf 未乘 100，容差为 **0.05 个组件/叶**。
- 判定为 `abs(reproduced_mean - paper_mean) <= tolerance`。样本标准差只展示；没有逐 seed 参考值或逐 seed 判定。默认值和判定逻辑均未改。
- Matched recall 的组件 IoU 阈值是 0.10，与四份发布方法的评估配置一致；它不等同于后续补充分析的一对一严格匹配定义。
- 原 docstring 记载重推理的末位变化约 0.01–0.02；README 保留 Supplementary Section S5 的末位差异说明，并将它与可配置的 0.05 验收容差分开说明。

### 论文事实逐条核对

| 事实 | 核查结果 |
| --- | --- |
| 707 ROI / 283 照片 | 两版 README 均一致。 |
| 503 / 104 / 100，按照片划分 | 一致；同一照片的叶片不跨集合。 |
| 60 px/mm | 两版 README 和原图工作流常量一致。 |
| 标签 0/1/2/3 | 数字与含义一致。中文原文使用“抗病型病斑”，提示词用“抗性型病斑”；按形态类别理解无数字冲突，术语是否统一留待作者决定，未自动改。 |
| 同类组件小于 24 像素删除，恰为 24 保留 | `leaf_grading.py` 使用 `area < 24` 排除；同类八连通组件分别测量。 |
| 不设物理面积下限 | 分级测量没有 mm² 下限，已在两版 README 明写。 |
| Dice / Small survival 基于未过滤掩膜 | 复现脚本读原始评估输出，原图工作流也保留原始掩膜；已明确 Table 7 分割指标与报告过滤口径的区别。 |
| seeds 100/101/102 | 两版 README、复现脚本和评估启动参数一致。 |
| 140 epoch × 2000 crops | 四份发布方法训练 YAML 和两版 README 一致。 |
| 按验证集 lesion Dice 选 best EMA | README 和训练代码的验证选优/EMA 使用方式一致。 |
| PyTorch 2.5.1 + CUDA 12.1 | README 记录一致。这是历史实验环境说明，不代表本机当前环境或锁定依赖；未改依赖文件。 |
| 对比模型权重不发布 | 索引只为 `hiblastnet/gcnet_bc/gcnet_lcjs/gcnet` 列发布权重，基线对比模型和其余消融没有发布路径，README 说法一致。 |

## 权重 SHA-256 对照

来源：`D:/Code/Writing/BlastSegPaper/deliverables/HiBlastNet_Weights.zip`，以及同名解压目录的 `manifest.json` 和 12 个文件。逐一流式读取 ZIP 内 checkpoint 的字节计算 SHA-256，没有反序列化、转换或重存权重。ZIP 内文件大小也与清单一致。

**论文对照来源是本次提示词列出的 Dataset S1/S2 记录。** 另外六个文件没有在提示词中提供论文哈希，因此只核对发布清单与实际文件，不虚称“已与全部论文哈希核对”。

| 文件 | 实际 SHA-256 | ZIP / manifest / 解压文件 | 提示词论文哈希 |
| --- | --- | --- | --- |
| `current_bc_lcjs_seed100.pth` | `9d3f5da2ef7117a7a679455498c4acb2d92ee61f3a488af4491f07911659bb10` | 一致 | 一致 |
| `current_bc_lcjs_seed101.pth` | `d828b1d41749b01c4bd674da2d342e1130d02c1d4318cac07b014db1a5c62998` | 一致 | 一致 |
| `current_bc_lcjs_seed102.pth` | `9b49f3dd769db333f80eb0ea229f12eb419ab5ffd793756d3057976ee79c17ad` | 一致 | 一致 |
| `current_bc_control_seed100.pth` | `9eda19af56462e83a41df0be845c0b6625f754a55d3f31bb2f122f9f631a91c1` | 一致 | 一致 |
| `current_bc_control_seed101.pth` | `47e2e24636e3863cb97bfed93aa86b7ab49d72e4387ffb332ca4fc7780c50d64` | 一致 | 一致 |
| `current_bc_control_seed102.pth` | `07d5e3a4f3ab9f57f1b6286b36249bbc025fb1efa7877dfe696ff505a167457b` | 一致 | 一致 |
| `current_b_lcjs_seed100.pth` | `d59bcf43e308619c91d6f39c995a60b2227fbbd90ff8ec76ba35d05ef81f7e95` | 一致 | 提示词未列出，未作独立论文哈希核对 |
| `current_b_lcjs_seed101.pth` | `7ee214cdc9be93e9b92eb77fba63927c57cdce86dab0c2d89ba8de19f0fe0684` | 一致 | 提示词未列出，未作独立论文哈希核对 |
| `current_b_lcjs_seed102.pth` | `614b90310b076d05fee249c60e4bd5cedd9a6207f139377027a1b484ae0c297d` | 一致 | 提示词未列出，未作独立论文哈希核对 |
| `current_b_plain_seed100.pth` | `9a11a5553d76307c48e4cb6d1404e13c5124d0764d58f6f5a79f774996d6a574` | 一致 | 提示词未列出，未作独立论文哈希核对 |
| `current_b_plain_seed101.pth` | `c2d8f0b70123cfcfae6a40077f57c87c31a4327631f276c68f8e65a294006d05` | 一致 | 提示词未列出，未作独立论文哈希核对 |
| `current_b_plain_seed102.pth` | `94544aeab444b3c939c7778731b72e2c2acf2d6755705216a1743f2c87e77087` | 一致 | 提示词未列出，未作独立论文哈希核对 |

这些文件名和摘要已完整列入两版 README 的权重章节（可展开表格）。如需用户自行复查本地下载文件，可在项目 PowerShell 中执行：

```powershell
Get-FileHash -LiteralPath 'weights/current_bc_lcjs_seed100.pth' -Algorithm SHA256
Get-ChildItem -LiteralPath 'weights' -Filter '*.pth' | Get-FileHash -Algorithm SHA256
```

本次核查的是已有本地发布包，**没有证明网盘当前实际返回的包就是该本地文件**；须下载后再核对。

## 下载链接核对与修复

方法：无登录、无预置 Cookie 的 HTTP GET，响应体最多读取 65,536 字节；记录状态码、跳转、页面标题。不提交提取码、不登录、不下载大型文件。

后续浏览器核对：在用户要求的内置浏览器进入 `https://onedrive.live.com/my`，分别打开 `blastleaf.zip` 和 `HiBlastNet_Weights.zip` 的“管理访问权限 → 链接”。每个文件各有 1 条“拥有链接的人员都可查看”的链接，地址与原 main 未提交 README 中较新的 OneDrive 地址一致。数据集链接设置中未设到期日期。两个当前地址均能打开对应 ZIP 的预览页面并显示“下载”按钮；页面提示 ZIP 无法预览，不是此前“无法访问此文档”的错误。

以上自动化浏览器核对基于已登录账号，没有独立完成未登录完整下载或远端包哈希核对。随后用户在另一台电脑测试这两个分享地址并反馈可以使用。本次未修改共享权限或重新上传文件，只替换 README 地址。

2026-11-20 后续自动化检查：使用无 Cookie、无登录凭据的独立 HTTP 会话，两个短链均重定向到对应 OneDrive 文件地址，随后均返回 **HTTP 403 Forbidden**；读取拒绝响应时还出现 SSL 错误，未取得 ZIP 字节。浏览器能看到正确文件和下载按钮，但下载控制调用超时。原始记录见交付包内 `anonymous_onedrive_check.json`。这些是自动化检测的局限，不能单凭它们判定分享失效；用户随后提供的跨电脑可用性测试是本报告当前结论的依据。未独立证明网盘包与本地发布包哈希一致。

测试时间：分支链接 2026-11-19 23:37 America/Denver；主工作树链接 23:36（对应 UTC 2026-11-20）。原始 JSON 的时间字段虽然命名为 `tested_at_utc`，实际保存的是带 `-07:00` 偏移的时间戳；此处已按偏移解释。

| 资源 | 当前分支 README 中的完整 URL | 提取码 | 已有测试结果 | 文件下载结论 |
| --- | --- | --- | --- | --- |
| 数据集 / OneDrive | [分享入口](https://1drv.ms/u/c/1495c176035c0989/IQCrwwXhApP_TpZNFzTIXQelAe2YA5BwKQ68frvLGekOpbQ?e=dtETWr) | — | 管理页为持有链接可查看；用户另一台电脑测试可用；自动化请求曾 HTTP 403 | 用户可用性测试通过；下载包哈希未独立核对 |
| 数据集 / 百度 | [分享入口](https://pan.baidu.com/s/1K6AEge89TWjPSs5MAovVrg?pwd=p1p6) | p1p6 | HTTP 200，提取码页，未验证取文件 | 需人工测试 |
| 30 张原图 / OneDrive | [分享入口](https://1drv.ms/u/c/1495c176035c0989/IQBtNABm1JC3SIWQSa1nlXPRAYTeY_4OhdFh1d5rHNjfLeA?e=g9yzBJ) | — | HTTP 200，动态应用入口，未验证取文件 | 需人工测试 |
| 30 张原图 / 百度 | [分享入口](https://pan.baidu.com/s/1EUgH1kOioeeU1JFz02Q_cw?pwd=ek4a) | ek4a | HTTP 200，提取码页，未验证取文件 | 需人工测试 |
| 权重 / OneDrive | [分享入口](https://1drv.ms/u/c/1495c176035c0989/IQC_DSvX2FfxRrrIhwz_O7L-AfOekbesJ90LgyT1dyQLT-o?e=HRmyBq) | — | 管理页为持有链接可查看；用户另一台电脑测试可用；自动化请求曾 HTTP 403 | 用户可用性测试通过；下载包哈希未独立核对 |
| 权重 / 百度 | [分享入口](https://pan.baidu.com/s/1EQbfUBzC3oSwfghtue5rhA?pwd=9zgt) | 9zgt | HTTP 200，提取码页，未验证取文件 | 需人工测试 |

OneDrive 的 HTTP 200 不能证明分享有效。初次检查的数据集和权重旧入口返回 Error 页面，用户也报告了旧数据集地址“无法访问此文档”的错误；这两个地址已从双语 README 移除。原图入口仅验证到通用动态页面。百度只到提取码页面，没有验证提取码和实际下载。

### 原 main 工作树的待提交链接差异（历史与当前状态）

首次检查时，分支基于已提交的 `f752635`，与原 main 未提交 README 有以下 4 项差异。下表保留修复前的地址作为审查记录；本次已采用右列两个 OneDrive 地址。百度两项仍保留分支原地址，尚待核对提取码和文件。

| 资源 | f752635 初次检查版本（历史） | 原 main 工作树的未提交版本 |
| --- | --- | --- |
| 数据集 / OneDrive | [分支链接](https://1drv.ms/u/c/1495c176035c0989/IQC3HxyG_ktHSpHlq_E-5vmXAYjrLFhmseHouaPyHpjrC-4?e=SkNipz) | [主工作树未提交链接](https://1drv.ms/u/c/1495c176035c0989/IQCrwwXhApP_TpZNFzTIXQelAe2YA5BwKQ68frvLGekOpbQ?e=dtETWr) |
| 数据集 / 百度 | [分支链接](https://pan.baidu.com/s/1K6AEge89TWjPSs5MAovVrg?pwd=p1p6) | [主工作树未提交链接](https://pan.baidu.com/s/1NVcVG6-oodVIuDr7NFXPbg?pwd=b1ia) |
| 权重 / OneDrive | [分支链接](https://1drv.ms/u/c/1495c176035c0989/IQBf-shaXSRVTaecn952a2AjAYvkrAHUWIZh25kWocqEaow?e=8L8BUk) | [主工作树未提交链接](https://1drv.ms/u/c/1495c176035c0989/IQC_DSvX2FfxRrrIhwz_O7L-AfOekbesJ90LgyT1dyQLT-o?e=HRmyBq) |
| 权重 / 百度 | [分支链接](https://pan.baidu.com/s/1EQbfUBzC3oSwfghtue5rhA?pwd=9zgt) | [主工作树未提交链接](https://pan.baidu.com/s/1TgPxlUOWIozmfnboUXg0Yw?pwd=i6jd) |

当前双语 README 的数据集和权重 OneDrive 地址已与原 main 的较新地址一致。30 张原图的两个 URL 没有分歧。百度新旧两组地址均没有经过实际取文件验证，因此未自动替换。

## 已修复：干净克隆缺失数据加载代码

修复前，基线提交 `f752635` 和对齐分支均未跟踪以下文件；现已从原 main 工作树逐字节恢复，没有改写算法：

- `blastseg/data/__init__.py`
- `blastseg/data/full_leaf.py`
- `blastseg/data/full_leaf_metadata.py`
- `blastseg/data/full_leaf_sampling.py`

修复前的证据：

- `git ls-files blastseg/data` 返回空。
- `git ls-files --others --ignored --exclude-standard -- 'blastseg/data/*.py'` 在原工作树列出以上四个文件。
- `git check-ignore -v blastseg/data/full_leaf.py` 指向 `.gitignore:5:data/`。
- 干净 worktree 没有 `blastseg/data/`，而 `blastseg/train.py`、`blastseg/test.py`、`blastseg/full_leaf_inference.py` 需要从该模块导入。
- 后续只读实测：在 `blastseg` 环境中分别导入 `blastseg.data`、`blastseg.train`、`blastseg.test` 和 `blastseg.full_leaf_inference`，四者均报 `ModuleNotFoundError: No module named 'blastseg.data'`。`scripts/run_photo_pipeline.py:26` 导入推理模块，原图分割评级流程也受此影响。
- 中文 README 早已有的链接 `blastseg/data/full_leaf.py` 在干净检出中失效。

上述问题原先阻断训练、评估及预测。用户授权修复后，已将 `data/` 改为 `/data/` 并恢复 4 个文件，逐文件 SHA-256 与原 main 一致。根目录 `data/blastleaf/` 仍被忽略，4 个源文件不再被误忽略，也未加入任何权重或数据资源。

验证使用干净源码快照：只复制 `git ls-files` 列出的文件及 4 个待提交源码，共 98 个文件；不含 `.git`、被忽略的本地源码、数据和权重，每个快照文件的哈希均与本工作树一致。运行时数据和发布权重通过显式外部路径提供。

- `blastseg.data/train/test/full_leaf_inference` 均能从快照导入；训练、评估、单叶预测、原图流程的 `--help` 全部返回 0。
- CPU 对 `685.jpg` 跑通标定、整叶提取、分割和评级，2 片叶等级为 `[0, 0]`；6 张 ROI/掩膜/叠加图及测量结果与此前验证输出完全一致。
- CPU 对 T1 第 1 片叶运行单叶预测，15027 × 675 掩膜与此前输出逐像素差异为 0，覆盖了有病斑的输入。
- 结果及完整源文件哈希清单见同级 `data_restore_validation_20261120T072506Z/validation_results.json` 和 `source_manifest.json`。没有执行完整训练或 12 个 checkpoint 的 Table 7 重评估。

## 验证与完整 diff 摘要

| 文件 | 本轮变更 |
| --- | --- |
| `README.md` | 当前题目、Citation 题目、Table 7 引用、12 个 SHA-256、容差单位/均值判定说明、未过滤分割指标与报告过滤的区别，以及数据集/权重两个 OneDrive 地址。代码块保持原样。 |
| `README.zh-CN.md` | 与英文版同步的题目、表号、哈希和统计口径。 |
| `scripts/check_reproduction.py` | 仅 Table 3 文案及 `PAPER_TABLE3` 符号同步改名；40 个数字、0.05 默认容差、读结果/计算均值/样本 SD/判定逻辑不变。 |
| `.gitignore` | 将资源忽略规则 `data/` 限定为根目录 `/data/`，避免忽略源码包。 |
| `blastseg/data/*.py`（4 个文件） | 从原 main 原样恢复并纳入版本控制，算法未修改。 |
| `PAPER_ALIGNMENT_REPORT.md` | 本报告。 |

已完成的验证：

1. 逐项断言 40 个参考数字与提示词一致，12 个发布文件名与索引一致。
2. 脚本文本差异严格限定为表号及同名引用替换，无算法变化；语法解析通过。
3. 两版 README 各含 12 个完整哈希，题目均已同步，许可证段落及原命令代码块与基线一致。后续修复只替换两个获准的 OneDrive 地址；其余四个下载链接均与基线一致，双语六个入口逐项相同。
4. `check_reproduction.py --help` 正常并显示 Table 7。
5. `--no-run` 正确报告缺少 12 次评估的结果并返回 1；这是预期的缺结果行为，**不是复现通过**，未触发模型评估。
6. 源码恢复后，原中文 README 中指向 `blastseg/data/full_leaf.py` 的失效路径也已恢复。
7. 提交前检查 diff 的文件范围、空白、许可证/配置/源算法未变，以及原主工作目录的两份 README 字节未变。新增源码保留原换行，Git 空白检查允许 Windows CRLF（`cr-at-eol`），其余默认空白规则保留。
8. 干净源码快照的 4 核心模块、4 命令入口及上述原图、单叶预测均通过；输出与旧验证结果一致。

完整补丁另交付为本 worktree 同级的 `PAPER_ALIGNMENT_FULL.diff`，不是只有 diffstat。验证记录、链接原始结果及哈希结果也放在同级目录；不将权重、原图或测试输出加入 Git。

## 需要作者处理或确认

1. **完整论文数值复现**：遗漏源码已修复，实际预测通过；全部 12 个 checkpoint 的 Table 7 重评估仍未执行。
2. **下载包核对**：两个 OneDrive 地址已由用户在另一台电脑确认可用，不再列为链接故障。下载包哈希未独立核对；百度两组新旧地址及提取码尚待选择和验证。
3. 若要求“全部 12 个哈希逐一与论文资料核对”，需提供 Baseline 和 Baseline + LCJS 的 6 个论文哈希；本轮已完整验证发布包自身一致性。
4. 中文“抗病型/抗性型”的术语取舍由作者决定；编码与形态含义没有发现矛盾。
5. 原 main 中已存在的未提交 README 修改仍留原处，包括英文 clone URL 的 `YOUR_USERNAME` 占位符及下载/解压说明差异。本分支使用已提交的真实 clone URL。审核时不要把原工作树差异误当本轮修改。
6. 用户现已授权推送 GitHub `main`，仅发布已审阅的修复；tag、Release 和 Zenodo 归档仍未执行。

## 可选发布建议（未执行）

完成上述阻塞项和实际下载验证后，可考虑标记 `v1.0-submission` 并创建 Release，再按项目需要归档到 Zenodo 生成 DOI。没有预先生成 DOI 或改变引用状态。

建议的 Release 说明草稿：

> Submission snapshot for “HiBlastNet: a scale-preserving segmentation network for small rice blast lesions in quantitative measurement and severity grading”.
>
> Includes training/evaluation code, bilingual usage instructions, original reference-board photo examples, and the raw-photo grading workflow. The four released configurations each provide seeds 100, 101, and 102. Checkpoint SHA-256 checksums are listed in the README. Reproduction references use the original unfiltered predictions in Table 7; small last-digit differences may occur on re-inference (Supplementary Section S5).
>
> Data and weights are distributed separately through the documented links. Existing code, third-party, dataset and weight licenses continue to apply. The paper is in preparation.

该草稿仅供作者决定。创建 tag、Release、Zenodo 归档均未执行。
