# 06 · 约束解码（让模型只输出合法内容）

## 一句话总结

给模型的每一步输出套一个"合法性过滤器"：凡是会让结果变得不合法（不符合 JSON /
不符合正则）的候选字，直接把它的得分压成负无穷——模型只能从没被封杀的字里选。

## 生活化类比

考驾照的**科目一答题卡**：你可以随便想，但填涂只能在 A/B/C/D 四个圆圈里。
约束解码就是在每个生成步骤动态算出"现在哪几个圆圈是合法的"，然后把笔按在合法范围内。

为什么需要它：让模型输出 JSON 给程序消费时，自由生成的 JSON 经常会少个引号、
多个逗号。约束解码从机制上**保证**输出是合法 JSON，下游解析永远不会失败。

## 它是怎么工作的

每一步解码：

```
模型吐出 logits（每个候选字的得分）
        │
        ▼
grammar.allowed_bytes(目前已生成的文本)
        │  例如当前是 {"nam ──▶ 合法下一字节 = 字母 / " / \ 等
        ▼
mask_logits：不合法的字 → 得分 -∞
        │
        ▼
sampler 正常采样（它永远选不到 -∞ 的字）
```

pagoda 内置两种约束（`src/grammar.rs`，零依赖）：

- **`Grammar::Regex`**：字节级正则（隐式 `^...$` 全匹配）。手写解析器编译成
  Thompson NFA，支持 `| ( ) [ ] . ? * + {m} ^ $ - \` 转义。每一步对 NFA 状态集做
  逆向可达性分析，算出"还有哪些字节能通往终点"。
- **`Grammar::Json`**：完整 JSON 值。一个下推自动机式的扫描器，
  知道"现在在对象里/数组里/字符串里/数字里"，给出每个位置合法的下一字节。

两个收尾细节：

- 输出已经完整时，EOS（结束符）会被保留为合法选项，让模型可以自然收尾。
- 走不下去了（死路）时直接以 `Stop` 结束，而不是硬采一个非法字符。

## 在 pagoda 里动手试试

```powershell
Invoke-RestMethod http://127.0.0.1:8080/generate -Method Post -ContentType "application/json" `
  -Body '{"text":"","sampling_params":{"max_tokens":32,"grammar":{"type":"json"}}}'

Invoke-RestMethod http://127.0.0.1:8080/generate -Method Post -ContentType "application/json" `
  -Body '{"text":"","sampling_params":{"max_tokens":8,"grammar":{"type":"regex","pattern":"\"[a-z]{3}\""}}}'
```

第二个例子的输出永远是形如 `"abc"` 的带引号三小写字母串——测试
`regex_grammar_produces_parseable_string_value` 锁死了这个行为。

## 一个重要限制

grammar 是**按字节**工作的：它假设"一个 token = 一个字节"。pagoda 内置的
`ByteTokenizer` 满足这个假设；换成 HuggingFace BPE 这类 tokenizer（一个 token 可能是
多个字节）时直接套用会封错候选。所以 pagoda 在 admission 时显式检查
`Tokenizer::is_byte_level()`，不满足就拒绝请求（`unsupported`），宁可报错也不静默出错。
真实系统（Outlines、SGLang）的做法是把 grammar 编译到 token 级——这是后续里程碑。

## 和 SGLang 的对照

| | SGLang | pagoda |
| --- | --- | --- |
| regex / JSON 约束 | ✅（集成 Outlines 等后端） | ✅ 零依赖自研 |
| 作用点 | logit mask | ✅ 相同 |
| 编译层级 | token 级 | 字节级（配字节 tokenizer） |

## 常见疑问

**Q：约束解码会让模型变笨吗？**
会有一点点——它限制了选择空间。pagoda 的 `mask_logits` 保留合法字的**原始得分**
（不是简单置 0），所以模型自己的偏好在合法范围内依然起决定作用。

**Q：能约束成 JSON Schema（指定字段）吗？**
`Grammar::Json` 只保证"是合法 JSON"。Schema 级约束可以先用 regex 拼简单模式，
完整的 JSON Schema 编译器在 roadmap 的 P2。