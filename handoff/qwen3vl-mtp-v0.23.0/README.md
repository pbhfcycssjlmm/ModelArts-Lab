# vLLM v0.23.0 双视频 MTP 直接替换交付

目标镜像没有 Git 元数据，因此本目录只保留可直接覆盖
`site-packages/vllm/` 的完整源码，不提供 `git apply` patch。

适用基线：

```text
vLLM v0.23.0
commit 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
```

必须一起替换 `replacement/` 中保持相对路径的五个文件：

```text
vllm/model_executor/models/qwen3_vl.py
vllm/v1/core/sched/scheduler.py
vllm/v1/worker/gpu/mm/encoder_runner.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu_model_runner.py
```

它们合并两项上游修复：

1. `d272418f459a`、`dd944845777b`：保留相邻视频间的视觉边界。
2. `3e6529cc0e`：修复 EAGLE/MTP encoder-cache 的 look-ahead 与 cache-miss 处理。

替换前备份目标文件；替换后对这五个文件执行 `python -m py_compile`，重启服务并使用同一双视频请求验证。

本次直接替换验证期间，暂时不要加载
`patch_qwen3vl_multivideo.py` 和 `patch_eagle_mm_encoder_cache.py` 两个 monkey patch；否则它们会再次
覆盖已替换的类方法，无法区分结果来自源码回移还是运行时 patch。

完整机制、镜像基线 hash、NPU monkey patch 的已知现象与验证标准见 [调查报告.md](调查报告.md)。
