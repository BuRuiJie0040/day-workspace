# Git Worktree 和 GitHub 使用说明

## 1. 当前分支是什么

当前仓库的默认分支是 `main`。

它的作用：

- 作为当前项目的主开发分支
- 保存当前目录 `D:\CODEX\day` 的正式提交历史
- 作为后续创建功能分支、测试分支、工作树的基础

常用命令：

```powershell
git branch
git switch main
git log --oneline
```

## 2. 新增加的内容分别是什么

### `.git/`

这是 Git 仓库的核心元数据目录，之前缺少它，所以 Codex 无法创建工作树。

它的作用：

- 记录提交历史
- 管理分支
- 支持 `git worktree`
- 支持后续绑定远程仓库

平时不要手动修改这个目录里的内容。

### `README.md`

这是仓库首页说明文件。

它的作用：

- 说明当前仓库为什么被初始化
- 记录仓库当前状态
- 提供最常用命令入口

你以后可以把它改成真正的项目说明。

### `docs/git-worktree-and-github-guide.md`

这是本说明文档。

它的作用：

- 解释分支是什么
- 解释工作树是什么
- 解释 GitHub CLI 和 Codex GitHub 插件分别做什么
- 给出典型使用方法

## 3. 什么是分支，怎么使用

分支可以理解为“同一个仓库里的独立开发线”。

它的典型用途：

- 在不影响 `main` 的情况下开发新功能
- 修复某个 bug
- 为不同任务分别保留代码状态

常见流程：

```powershell
git switch -c feature/login
```

这条命令会新建并切换到 `feature/login` 分支。

开发完成后可以查看状态：

```powershell
git status
git add .
git commit -m "Add login feature"
```

再切回主分支：

```powershell
git switch main
```

## 4. 什么是 worktree，怎么使用

`worktree` 的作用是：让同一个 Git 仓库在多个文件夹里同时打开不同分支。

这对 Codex 很有用，因为你可以：

- 在主目录保留稳定版本
- 在另一个目录让 Codex处理新任务
- 不需要反复切换和覆盖当前文件

例如，新建一个工作树：

```powershell
git worktree add ..\day-feature-demo -b feature/demo
```

它的含义：

- 在 `D:\CODEX\day-feature-demo` 创建一个新目录
- 同时创建一个新分支 `feature/demo`
- 这个新目录直接绑定到该分支

查看所有工作树：

```powershell
git worktree list
```

删除工作树：

```powershell
git worktree remove ..\day-feature-demo
git branch -D feature/demo
```

注意：

- 只有 Git 仓库才能使用 `worktree`
- 这也是你之前报错 `Not a git repository` 的根因

## 5. GitHub CLI `gh` 是什么，怎么使用

`gh` 是 GitHub 官方命令行工具。

它的作用：

- 登录 GitHub
- 创建 PR
- 查看 PR 和 issue
- 查询 Actions 状态
- 在终端里完成很多 GitHub 操作

当前状态：

- 已安装
- 还没有登录

先登录：

```powershell
gh auth login
```

登录后可用命令示例：

```powershell
gh auth status
gh repo view
gh pr list
gh pr create
gh issue list
```

如果这个仓库以后绑定了远程仓库，还可以直接创建 PR。

## 6. Codex GitHub 插件是什么，怎么使用

Codex GitHub 插件和 `gh` CLI 不完全一样。

它的作用：

- 让 Codex 直接读取 GitHub 仓库、PR、issue、检查状态
- 让我帮你总结 PR 改动、 review 意见、CI 问题
- 让我在已有分支已推送的前提下帮你创建 PR

当前状态：

- 插件能力已加载
- 但当前没有读到任何已安装账号或组织

这通常表示以下步骤还没完成：

- 你在 Codex 里连接 GitHub
- 你在 GitHub 页面里同意授权
- 你把 GitHub App 安装到个人账号或组织

完成后，你就可以直接让我做这些事：

- “帮我总结这个 PR 改了什么”
- “看一下这个 PR 为什么 CI 失败”
- “根据我当前分支生成 PR 标题和描述”
- “帮我列出这个仓库最近需要处理的 PR 和 issue”

## 7. 推荐使用顺序

建议按下面顺序使用：

1. 在本地目录里开发和提交代码
2. 用分支隔离不同任务
3. 用 `worktree` 给 Codex 开并行任务目录
4. 用 `gh auth login` 完成 GitHub CLI 登录
5. 在 Codex 里完成 GitHub App 授权安装
6. 再让我帮你做 PR、CI、review、issue 联动

## 8. 现在你可以直接用的命令

```powershell
git status
git branch
git worktree list
gh auth login
gh auth status
```
