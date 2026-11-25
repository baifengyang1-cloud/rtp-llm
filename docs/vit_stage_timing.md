# ViT 分阶段耗时日志

INFO 日志标记为 `vit_embedding_stage_timing`，默认全量记录，不受
`VIT_TRACE_SAMPLE_EVERY` 控制。每阶段记录 `phase=begin/end`，异常结束记录
`outcome=error` 和异常类型，不记录异常内容、图片 URL 或请求正文。

| stage | 范围 |
|---|---|
| `embedding_forward` | 调度器调用整个 `batched_embedding`，包含设备图片处理、视觉前向和输出整理，不含后续特征哈希与缓存完成 |
| `vision_forward` | Qwen3.5 dense/MoE 的实际 `self.visual(...)` 调用，不含图片设备处理与输出整理 |
| `feature_hash` | 单个 work item 的特征哈希，包含首次原生模块加载以及哈希函数内部已有的同步与传输 |
| `cache_complete` | embedding 缓存完成和 hash-key 缓存登记，不含前面的特征哈希 |

`vision_forward` 嵌套在 `embedding_forward` 内，两者不能相加。
`feature_hash` 和 `cache_complete` 每个 work item 分别记录。
已有哈希、延迟完成缓存或没有缓存 claim 时，记录 `outcome=skipped` 和 `reason`。

批次内共享 `batch_id`、`request_ids`、`image_ids`；哈希与缓存阶段额外记录当前
`request_id`。`image_ids` 是 URL 的不可逆摘要，与现有图片耗时日志一致。
批次 ID 含 PID 和单调时钟值，重试时生成新 ID。批次外强制完成缓存时
`batch_id=null`，仍可通过 `request_id` 对齐。

## 计时口径

默认 `timing_mode=host`，`host_ms` 和 `elapsed_ms` 是主机调用耗时；不会新增
GPU/PPU 同步。异步设备工作可能在后续特征哈希阶段才被等待，不能将 host 计时
直接解释为 kernel 执行时间。

需要进一步区分视觉计算与哈希等待时，在 ViT 进程启动前设置：

```ini
RTP_LLM_VIT_STAGE_TIMING_SYNC=1
```

该开关仅在 `vision_forward` 前后同步当前设备；没有可用设备时使用 host 计时。
使用 Torch CUDA 接口，包含兼容该接口的 PPU 后端。

- `pre_sync_ms`：视觉前向开始前排空已有设备工作的等待，单独记录，不计入本次 `elapsed_ms`。
- `host_ms`：视觉前向的主机调用耗时。
- `device_sync_ms`：视觉前向返回后等待设备完成的耗时。
- `elapsed_ms`：本次主机调用和末尾同步的总耗时。

同步模式会改变异步重叠和请求耗时，供临时诊断使用；设备级同步时间也可能受到
同进程其他 stream 工作影响，并非纯 kernel 性能测量。

示意日志（数值仅为格式示例）：

```json
{"stage":"feature_hash","phase":"end","batch_id":"123:456","pid":123,"request_id":42,"request_ids":[42],"image_ids":["opaque-image-id"],"timing_mode":"host","outcome":"ok","host_ms":12.3,"elapsed_ms":12.3,"pre_sync_ms":0.0,"device_sync_ms":0.0}
```

这些日志需要包含修改后的 ViT 镜像部署后才会在线上出现。
