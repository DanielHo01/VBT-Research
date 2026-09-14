# 分支治理策略 (Trunk-Based Development)

> 生效日期：v0.1.0-baseline 起
> 适用范围：VBT-Research 仓库所有分支活动

---

## 核心原则

**main 是唯一长寿命分支。** 所有新工作通过短期 feature 分支 + Pull Request 合并回 main，禁止长期独立的开发分支。

---

## 分支类型

| 分支类型 | 命名规范 | 生命周期 | 合并方式 |
| --- | --- | --- | --- |
| 主干 | `main` | 永久 | 受分支保护（PR + CI + review） |
| 功能 | `feat/<scope>` | < 5 天 | Squash merge 到 main |
| 修复 | `fix/<scope>` | < 3 天 | Squash merge 到 main |
| 实验 | `exp/<topic>` | < 7 天 | 完成后归档到 `_archive/` 或 Squash merge |
| 留出集 / 归档 | `arena/<id>-<topic>` | 永久 | 只读，不合并 |

`<scope>` 命名约定：kebab-case，描述具体改动领域，例如：

- `feat/vbtcore-cpp-port`
- `feat/android-jni-bridge`
- `fix/kalman-numerical-drift`
- `exp/int8-quantization`

---

## 提交规范 (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

**type 必须从下表选**：

| type | 用途 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | bug 修复 |
| `docs` | 仅文档改动 |
| `test` | 仅测试改动 |
| `refactor` | 既不修 bug 也不加功能 |
| `perf` | 性能优化 |
| `chore` | 工具链 / 依赖 / 配置变更 |
| `merge` | 分支合并（squash 后用） |

**subject 规则**：

- 50 字符内
- 祈使句（"add" 而非 "added"）
- 不大写首字母
- 不加句末标点

**body**（可选）：说明 *为什么* 改，不写 *怎么改* 的细节。

---

## Pull Request 规范

每个 PR 必须满足：

- [ ] 标题遵循 Conventional Commits
- [ ] 描述包含：背景、改动、影响范围、测试方式
- [ ] 通过 CI（`python tests/run_all_tests.py` + lint）
- [ ] 至少 1 人 review（单人项目：自我 review + 截图/日志佐证）
- [ ] Squash merge（保留 main 历史干净）
- [ ] 删除源分支

### PR 模板

```markdown
## 背景
<为什么需要这个改动？引用 issue / docs>

## 改动
<主要技术改动清单>

## 影响范围
<API 变更、依赖变更、性能影响>

## 测试
<本地验证步骤 + 截图/日志>

## 风险与回退
<可能的副作用 + 如何回退>
```

---

## 受保护分支（main）

GitHub 端配置：

- ✅ Require pull request before merging
- ✅ Require 1+ approval
- ✅ Require status checks (CI) to pass
- ✅ Require branches to be up to date
- ❌ 禁止 force push
- ❌ 禁止直接 push（必须 PR）

---

## Tag 规范

| Tag 格式 | 用途 | 示例 |
| --- | --- | --- |
| `v<MAJOR>.<MINOR>.<PATCH>` | 正式版本 | `v1.0.0` |
| `v<MAJOR>.<MINOR>.<PATCH>-baseline` | 大改前快照（基线回溯点） | `v0.1.0-baseline` |
| `v<MAJOR>.<MINOR>.<PATCH>-rc<N>` | 发布候选 | `v1.0.0-rc1` |
| `v<MAJOR>.<MINOR>.<PATCH>-<descriptor>` | 阶段里程碑 | `v1.0.0-android-demo` |

**版本号语义**：

- MAJOR：API 不兼容
- MINOR：向下兼容的功能新增
- PATCH：向下兼容的 bug 修复

---

## 工作流示例

### 新功能（典型）

```bash
git switch main
git pull --rebase
git switch -c feat/android-jni-bridge
# 改代码 + 写测试
python tests/run_all_tests.py   # 必须全过
git push -u origin HEAD
# 在 GitHub 开 PR，标题：feat(android): JNI bridge + Compose UI skeleton
# 1 reviewer approve + CI pass → Squash merge → 删除源分支
```

### 紧急修复

```bash
git switch -c fix/kalman-numerical-drift main
# 改代码 + 加 regression test
git push -u origin HEAD
# PR 标题：fix(kalman): 数值漂移超阈值的回归测试
# 合并后必须立即 release patch 版本
```

### 实验性工作

```bash
git switch -c exp/int8-quantization main
# 实验代码（标注 EXPERIMENTAL）
git push -u origin HEAD
# 实验成功 → 转 PR 合并
# 实验失败 → 归档到 _archive/<date>-<topic>/
```

---

## 禁止事项

- ❌ 直接 push 到 main（必须 PR）
- ❌ 长期 feature 分支（> 5 天必须拆或并）
- ❌ 在 main 上 force push
- ❌ 提交未通过测试的代码
- ❌ 跳过 CI 直接合并
- ❌ 混合多个无关改动的单次提交

---

## 历史归档

`feat/phase0-pure-detection`、`feat/vbtcore-v5-refactor` 已于 v0.1.0-baseline 之前 squash merge 到 main。原分支已删除。相关文档归档于：

- `docs/archive/STEP2_ARCHITECTURE.md`
- `docs/archive/DATA_DD_20260912.md`
- `docs/archive/SESSION_HANDOFF_20260911.md`

---

## 参考

- [Trunk Based Development](https://trunkbaseddevelopment.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- 内部文档：`docs/ARCHITECTURE.md`

---

*制定：AI 助手（受 Daniel 委托） | 生效：v0.1.0-baseline 起*
