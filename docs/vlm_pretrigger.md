# VLM pretrigger user guide

VLM pretrigger lets your application start image/video preprocessing and encoder
computation as soon as media is available, while it finishes preparing the prompt.
The later inference request can reuse ongoing or completed work on the same ViT
worker. Reuse is best effort: if the work is unavailable, normal inference computes
what it needs.

**HTTP 200 means the encoder work was admitted, or there was no media to submit.**
It does not mean embeddings are ready. Pretrigger returns no generated answer;
your application sends a separate normal inference request to get one.

## Before you start

Use a deployed RTP-LLM frontend configured for **remote ViT**, with one worker
process per ViT instance (`vit_server_count=1`). Multiple standalone ViT instances
may share a pool. Local ViT and multi-worker ViT proxy deployments are unsupported
for media pretrigger.

The deployed model must support the media you submit. The chat examples below
use Qwen3.5 image/video content formats. Other models may support different media
or reject media pretrigger. Use media URLs that the ViT service can access.

The examples assume a running service at `http://localhost:8000`. Replace that
address and the illustrative `https://example.com/...` URLs. Include your
service's normal authentication and routing headers on both requests.

## Choose the request interface

| Interface | Endpoint | Enable pretrigger |
| --- | --- | --- |
| Chat | `POST /chat/completions` or `POST /v1/chat/completions` | `extra_configs.pretrigger_scheme="encoder"` |
| Raw/native | `POST /` | `generate_config.pretrigger_scheme="encoder"` |

`pretrigger_scheme` accepts exactly two case-sensitive strings:

- `"encoder"`: submit encoder work without generating text. Prompt text or
  conversation history may be incomplete.
- `"disable"`, or an omitted field: run conventional inference with its normal
  input validation. Supply the complete inference input.

Every other value, including `null`, booleans, numbers, and `"Encoder"`, returns
HTTP 400. This validation also applies to requests without media.

Use the same interface for pretrigger and subsequent inference to preserve media
identity.

## Quick start: chat

### 1. Submit the media early

As soon as the image URL and preprocessing options are known, send a media-only
request. The prompt can be assembled separately while this request is in flight.

```bash
curl -sS -i http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "messages": [{
    "role": "user",
    "content": [{
      "type": "image_url",
      "image_url": {"url": "https://example.com/image.jpg"}
    }]
  }],
  "extra_configs": {"pretrigger_scheme": "encoder"}
}
JSON
```

A successful non-streaming response looks like this (`created` is illustrative):

```json
{
  "id": "chat-",
  "object": "chat.completion",
  "created": 1789689600,
  "model": "",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "partial": false},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}
}
```

Empty `message.content` is omitted. Zero usage describes language-model token
accounting for this request; it does not indicate that encoder computation is
free. The `id` and `model` shown are current response defaults, not a job handle
or a way to retrieve embeddings. There is no completion-polling API.

### 2. Send normal inference when the prompt is ready

Use the same media URL and preprocessing options, add the final text/history,
and **omit the scheme or set it to `"disable"`**:

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
      {"type": "text", "text": "Describe the objects in this image."}
    ]
  }],
  "max_tokens": 128
}
JSON
```

This request generates the answer. Leaving `"encoder"` enabled would submit
encoder work again and return another empty completion.

There is no need to wait for encoder completion before inference. Your application
can proceed with normal inference even if pretrigger fails or its acknowledgement
has not arrived. Matching requests can share work in either arrival order when
caching is enabled and they reach the same worker.

## Multiple images, video, and preprocessing

Put multiple media parts in the chat content list. For example, this Qwen3.5
request pretriggers two images and a video:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/front.jpg"}},
      {"type": "image_url", "image_url": {"url": "https://example.com/back.jpg"}},
      {
        "type": "video_url",
        "video_url": {"url": "https://example.com/clip.mp4"},
        "preprocess_config": {"fps": 2}
      }
    ]
  }],
  "extra_configs": {
    "pretrigger_scheme": "encoder",
    "min_pixels": 65536,
    "max_pixels": 1048576
  }
}
```

Media preprocessing settings remain effective in encoder mode, including
request-level settings in `extra_configs` and per-part `preprocess_config`
overrides. Keep their resolved values the same in the inference request.
Token-generation settings such as sampling and output token limits do not start
generation or require a complete tool conversation in encoder mode. Media
structure and response-shaping controls are still validated.

One HTTP response covers the whole media list. HTTP 200 means every item was
admitted or attached to reusable work. A failure may occur after some items were
submitted; those items are not rolled back.

## Raw/native requests

For native callers, send `images` (or its `urls` alias) to `/`. Encoder mode does
not require `prompt` or `text`:

```bash
curl -sS -i http://localhost:8000/ \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "images": ["https://example.com/image.jpg"],
  "generate_config": {"pretrigger_scheme": "encoder"}
}
JSON
```

The default successful response is:

```json
{"response": "", "finished": true, "aux_info": {}}
```

For subsequent inference, send the full native prompt with its usual model-specific
media placeholders, the same media and preprocessing settings, and the scheme
omitted or disabled. Follow your deployed model's normal native prompt format.

### Batches and multiple response slots

Nested media groups without prompt/text fields select a batch, one response per
group. Empty groups retain their response slot:

```json
{
  "images": [
    ["https://example.com/front.jpg", "https://example.com/back.jpg"],
    []
  ],
  "generate_config": {"pretrigger_scheme": "encoder"}
}
```

Response:

```json
{
  "response_batch": [
    {"response": "", "finished": true, "aux_info": {}},
    {"response": "", "finished": true, "aux_info": {}}
  ]
}
```

If you already have a `prompt_batch`, its presence selects the batch wrapper,
including a singleton batch. A no-media request with `prompt_batch=["", ""]`
still returns two entries. A list-valued `prompt` or `text` alone does not select
that wrapper. These batch rules apply to `/`; this feature does not add encoder
dispatch to `/batch_infer` or `/v1/batch/chat/completions`.

`generate_config.num_return_sequences` controls empty response slots per input;
it does not control the number of submitted media items:

| Value | Response per input, with default auxiliary info enabled |
| --- | --- |
| Omitted or `0` | `{"response":"","finished":true,"aux_info":{}}` |
| `1` | `{"response":[""],"finished":true,"aux_info":[{}]}` |
| `2` | `{"response":["",""],"finished":true,"aux_info":[{},{}]}` |

Setting `generate_config.aux_info=false` makes list-mode `aux_info` an empty list.
Scalar mode always has an empty dictionary. No engine measurements are fabricated.

For native configuration, `generation_config` is an alias. If both containers
are supplied, `generate_config` wins; recognized top-level fields override the
nested values. Prefer one `generate_config` object to avoid conflicting settings.

## Streaming responses

Set top-level `"stream": true` on a chat or raw encoder request to use SSE.
Use `curl -N` to display events without buffering. Admission completes before
HTTP headers are sent, so admission errors retain their HTTP error status.

Chat emits exactly one terminal chunk, with an empty delta and zero usage:

```text
data: {"id":"chat","object":"chat.completion.chunk","created":1789689600,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"total_tokens":0,"completion_tokens":0}}

```

The timestamp is illustrative. SSE events end with `\r\n\r\n`. Top-level
`stream=true` closes after the terminal event, **without a `[DONE]` or `[done]`
sentinel**, for both chat and raw requests. A raw event contains the same
scalar/list/batch response body as its non-streaming equivalent.

| Raw streaming control | HTTP behavior |
| --- | --- |
| Top-level `stream=true` | SSE with `data: ` prefix, one terminal event, no sentinel |
| `yield_generator=true`, with `stream` absent/false | SSE with `data:` prefix, one terminal event, then `data:[done]` |
| Only `generate_config.is_streaming=true` | JSON; this engine flag does not select HTTP SSE |

A top-level raw `yield_generator` overrides the nested value, including explicit
false. For this transport option only, the existing helper checks
`generation_config` before `generate_config` if both are present. Use one config
container for predictable behavior. Raw-only transport fields do not select
streaming on the chat interface.

Chat choice count defaults to one, or follows `n` when supplied. An effective beam
width other than one overrides `n`: use the last `variable_num_beams` entry when
there is more than one entry, otherwise `num_beams`. These controls allocate empty
response slots only. `extra_configs.num_return_sequences` alone does not select
chat choice count. The same cardinality applies to JSON and SSE responses.

## No media, errors, and retries

A valid encoder request without media returns the same empty HTTP 200 response
without contacting ViT. This allows callers to use one submission flow for
text-only and multimodal inputs. Malformed media is rejected rather than treated
as absent.

| HTTP status | Meaning | Caller action |
| --- | --- | --- |
| `200` | All media admitted/reused, or no media supplied | Continue preparing or sending normal inference |
| `400` | Invalid scheme, malformed media/options, or unsupported media extraction | Correct the request; check the deployed model's media support |
| `403` | Policy rejection | Check authorization/routing policy with the service owner |
| `429` | Capacity exhausted | Reduce speculative traffic; retry only if useful lead time remains |
| `503` | Unsupported topology or no available ViT worker | Check deployment/routing; normal inference may still be attempted |
| `504` | Routing/submission deadline expired | Treat admission as unknown; normal inference may still be attempted |
| `500` | Unexpected server error | Use service logs and request tracing to diagnose |

Errors use the normal `error_code`, `error_code_str`, and `message` envelope.
An invalid scheme's message identifies `pretrigger_scheme` and the allowed
`"disable"` / `"encoder"` values.

Pretrigger is optional preparation: its failure need not block normal inference.
Normal inference still has its own capacity and availability requirements.
If you retry pretrigger, you can resend the full media list; retained matching
work on the same worker can be reused. A timeout or lost connection does not prove
that nothing was admitted. Accepted work can continue after the caller disconnects,
and there is no pretrigger cancellation endpoint. Worker shutdown or background
failure can still discard accepted work.

## Reuse and latency expectations

To give pretrigger a useful lead, submit once the media and preprocessing options
are final, while other application work continues. Avoid waiting for it on the
critical path once the final inference request is ready.

Reuse depends on matching media identity and worker placement:

- Keep the exact URLs, media types, and resolved preprocessing settings. Different
  signed URLs for the same bytes may miss; a changed resize or frame-sampling
  configuration may also miss.
- Keep the media accessible and stable at its URL. Matching is not a content digest.
- Preserve the request interface: native media uses a default media type, while
  chat image/video parts use explicit types. Identical URLs across interfaces do
  not guarantee matching keys.
- Send normal inference soon enough that the work remains available. Cache
  eviction, worker replacement, or routing to another instance can prevent reuse.
- Extra media in the final request is allowed. For example, pretrigger A followed
  by inference with A+B can reuse A and compute B normally.

Pretrigger shares bounded ViT capacity and scheduling with normal inference.
There is no lower-priority queue or priority promotion in this version. Heavy
speculative traffic can delay normal inference or increase admission rejection.
Compare end-to-end latency with pretrigger on/off using matched requests and warm
kernels, and monitor encoder queue pressure and reuse. An HTTP 200 alone does not
measure saved latency.

## Service configuration and monitoring

For an existing remote-ViT deployment, confirm these settings with the service
owner. They supplement the deployment's model and routing configuration; setting
them alone does not create a working ViT service.

| Setting | CLI equivalent | Purpose |
| --- | --- | --- |
| `VIT_SEPARATION=2` | `--vit_separation 2` | Frontend uses remote ViT |
| `VIT_SERVER_COUNT=1` | `--vit_server_count 1` | Supported single-worker topology per ViT instance |
| `PRETRIGGER_TIMEOUT_MS=5000` | `--pretrigger_timeout_ms 5000` | Total routing/submission acknowledgement budget; must be positive |

The submission deadline is independent of the embedding computation timeout and
cache retention. Set the client's HTTP timeout to leave room for this submission
budget and network overhead; increasing it does not guarantee admission or reuse.

FlexLB routing is preferred for media affinity. When it is absent or unreachable
after the existing connection attempts, ViT-domain discovery may select an
instance, including from a multi-instance pool. Explicit policy or capacity
rejection is returned to the caller without discovery bypass.

Keep embedding caching enabled for work sharing and retained-result reuse. See
[ViT embedding cache configuration](vit_cache_capacity.md) for GPU/CPU budgets
and eviction behavior. Disabling both embedding-cache tiers removes persistent
reuse and cache-based sharing of ongoing work.

Useful kmonitor signals include `py_rtp_pretrigger_outcome_qps`,
`py_rtp_pretrigger_submission_rt_ms`, `py_rtp_vit_pretrigger_reuse_qps`, and
`py_rtp_vit_inference_wait_rt_ms`, alongside existing queue, cache, and admission
metrics. Static configuration is logged at startup; background failures are
reported through logs and metrics. Use ordinary request tracing for correlation,
not response IDs as result-lookup handles.
