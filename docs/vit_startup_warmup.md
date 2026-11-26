# ViT startup warmup and request deadlines

Standalone ViT workers now preload native compute/engine libraries and warm
Qwen35 vision inputs **before starting their HTTP/gRPC endpoints**. The callback
runs on the `mm-scheduler` thread, after binding its CUDA/PPU device, so serving
reuses the same thread's accelerator handles and first-use state. Every worker
process does this independently. A per-user, per-device file lock in `/tmp`
serializes local workers' warmup; no lock is used for serving. The proxy's
existing all-workers health probe therefore waits for every worker to finish.
Warmup failure/timeout aborts worker startup instead of advertising readiness.

Synthetic RGB images exercise configured pixel normalization/patch packing,
vision forward, positional IDs and the real native feature hash. If GPU JPEG
decoding is enabled, synthetic JPEGs also exercise its decode/resize path and
existing fallback. No media URLs, embedding-cache entries or request admission
slots are created. Outputs are released after each iteration. This does not
change image sizes or algorithms used for real requests.

The default warmup covers 13 common image shapes and four multi-image
compositions, twice each. Per-image pixel limits, the scheduler's media limit
and any model work budget are respected. Shapes outside this finite list can
still have first-use latency; this is not a guarantee to precompile every
possible resolution/composition. Models without a warmup-input provider preload
native libraries and log `model_has_no_warmup_inputs`.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `RTP_LLM_VIT_STARTUP_WARMUP` | `1` | Enable startup preload/warmup for standalone ViT workers; `0` is an explicit rollback. Embedded LLM-engine callers retain their existing startup behavior. |
| `RTP_LLM_VIT_WARMUP_TIMEOUT_S` | `600` | Independent startup budget, including waiting for the device warmup lock. Allow this in platform startup/liveness grace periods. |
| `RTP_LLM_VIT_WARMUP_REPEATS` | `2` | 1–10 iterations per composition. |
| `RTP_LLM_VIT_WARMUP_BATCHES` | built-in hot list | JSON array of batches of `[height,width]`, e.g. `[[[512,512]],[[800,800],[800,800]]]`. Dimensions must match the model patch/merge factor; at most 64 batches and 32 images per batch. |

Logs `vit_warmup_lock`, `vit_warmup_native`, `vit_warmup_batch` and
`vit_warmup_ready` identify the process, compositions, repeats and durations.
`vit_warmup_skipped` explains skipped budgets or unsupported model warmup.
Use `RTP_LLM_VIT_STAGE_TIMING_SYNC=0` for normal serving; warmup explicitly
synchronizes each iteration itself, independent of the diagnostic timing flag.

Async image work now carries its original monotonic owner deadline through
downloads, preprocessing and scheduler admission. Joining a shared cache entry
adds a separate owner's deadline; querying again as the same owner cannot
restart its budget. The last active owner's deadline governs shared work, and
last-owner cancellation invalidates queued GPU work. Deadline checks occur
before admission, during batch collection, before forward/OOM retries, and
before enqueueing a following chunk. Waiting callers observe owner changes
within 50ms. Synchronous preprocessing/cache waits also use remaining budget.
Expired forwards are not hashed or published through the scheduler, and log
`vit_scheduler_skip`. Already executing accelerator kernels cannot be forcibly
interrupted; they must finish before that executor can run another batch.

Roll out the new image in the **ViT zone** with startup grace sufficient for all
local workers, preserving existing GPU image-processing/pixel-cap settings.
Keep an old Ready instance during warmup when continuous availability is needed.
This change does not fix time spent after ViT in prefill/decode generation.
