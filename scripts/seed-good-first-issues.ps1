# 用 gh CLI 批量创建 good-first-issue（把 GOOD_FIRST_ISSUES.md 落成真 issue）
# 前置：gh auth login && gh repo set-default quantz8a/pagoda
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$tasks = @(
  @{ d='L1'; a='TTFT · Useful Life'; t='移植 Qwen2-VL 图像处理器'; w='参考 python/sglang/srt/multimodal/processors/qwen_vl.py，按 rs-mm/src/model/internvl.rs 实现 process_item/layout，用 Geometry::Grid([t,h,w])'; s='rs-mm 下 cargo test --offline，加 grid_split_matches_reference' },
  @{ d='L1'; a='TTFT'; t='移植 MiniCPM-V 图像处理器'; w='参考 minicpmv.py 的 slice 切分，layout 用 TokenPattern::Explicit 拼分隔符'; s='slice 长图 1 vs 多 slice token 数对得上' },
  @{ d='L1'; a='Useful Life'; t='移植 LLaVA-1.5 图像处理器'; w='最简单一条：单图 resize 到固定 grid，参考 llava.py'; s='feature.shape=[1,3,h,w]，normalize 值域校验' },
  @{ d='L2'; a='Useful Life'; t='common/decode 接真实 PNG 解码'; w='rs-mm/src/common/decode.rs 目前只有 BMP；接入零依赖 PNG(flate)，JPEG 留占位'; s='base64 小图解码后像素与参考一致' },
  @{ d='L1'; a='TTFT'; t='第 4 种调度策略 ShortestRemaining'; w='pagoda/src/engine.rs 的 SchedulePolicy 加一个变体'; s='长短混合队列断言出队顺序' },
  @{ d='L1'; a='TTFT · Token/Watt'; t='Radix 命中率单测扩展'; w='pagoda/src/radix_cache.rs 补分支/删除节点后命中率断言'; s='cargo test --offline' },
  @{ d='L2'; a='Token/Watt · Useful Life'; t='evict_lru 非叶节点回收策略'; w='pagoda/src/radix_cache.rs evict_lru 目前只回收叶子'; s='引用为 0 才可回收 + 单测' },
  @{ d='L1'; a='Useful Life'; t='OpenAI 端点契约回归测试'; w='pagoda/tests/system_tests.rs 锁死 /v1/chat/completions 与 /generate 形状'; s='cargo test --offline' },
  @{ d='L1'; a='Useful Life'; t='补一页 guide 文档'; w='pagoda/docs/guide/ 任选一个已实现机制，对齐 03-radix 风格'; s='markdown 渲染 + 示例可跑' },
  @{ d='L2'; a='Token/Watt · Useful Life'; t='pagoda-hf 接一个新后端'; w='照 pagoda-hf/src/llama.rs 加 RMSNorm+GQA+RoPE 模型(Qwen/Mistral 优先)'; s='加一条 e2e_*.rs' },
  @{ d='L1'; a='Revenue'; t='/stats 字段单测'; w='锁死 14 字段 + 4 派生 KPI 的存在与单调性'; s='cargo test --offline' },
  @{ d='L2'; a='Token/Watt'; t='Token/Watt 真实度代理增强'; w='把 avg_forward_per_output_token 迁成 decode_batch_factor/kv_pages_reused 加权口径'; s='字段语义注释 + 单测' }
)

foreach ($x in $tasks) {
  $body = "难度：$($x.d)`n驱动轴：$($x.a)`n从哪下手：$($x.w)`n自测：$($x.s)`n`n来自 GOOD_FIRST_ISSUES.md"
  gh issue create --title "[good-first-issue] $($x.t)" --label "good first issue" --body $body
  Write-Host "created: $($x.t)"
}