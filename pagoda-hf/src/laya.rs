// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Laya decision model (System-1): typed answers with calibrated
//! probabilities in ONE non-autoregressive forward pass.
//!
//! Port of the reference pipeline (`rl_agent_api.py` + `rl_common.py` from
//! the Laya repo) onto Candle: a ModernBERT encoder plus a decision head
//! (2-layer transformer over option markers, scorer, act head), with
//! post-hoc temperature calibration per question-type bucket.
//!
//! Question types (Jev-compatible request shape):
//! * `choice` — pick one of N options (softmax probabilities per option)
//! * `score`  — ordinal scale 0..k-1 (expected value under the distribution)
//! * `noul`   — probability that a statement holds (p of "true")
//!
//! The model never generates text: given a `state` and typed `questions`,
//! [`Laya::decide`] returns typed answers + confidences.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, ops::softmax_last_dim, Embedding, LayerNorm, Linear, Module,
    VarBuilder,
};
use candle_transformers::models::modernbert::{Config as MbConfig, ModernBert};

use crate::HfTokenizer;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `encoder/config.json` (ModernBERT). candle's `modernbert::Config` expects
/// flattened rope thetas, so we parse Laya's nested shape and convert.
#[derive(serde::Deserialize)]
struct EncoderConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    pad_token_id: u32,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    rope_parameters: RopeParameters,
}

#[derive(serde::Deserialize)]
struct RopeParameters {
    full_attention: RopeTheta,
    sliding_attention: RopeTheta,
}

#[derive(serde::Deserialize)]
struct RopeTheta {
    rope_theta: f64,
}

impl From<EncoderConfig> for MbConfig {
    fn from(c: EncoderConfig) -> Self {
        MbConfig {
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            intermediate_size: c.intermediate_size,
            max_position_embeddings: c.max_position_embeddings,
            layer_norm_eps: c.layer_norm_eps,
            pad_token_id: c.pad_token_id,
            global_attn_every_n_layers: c.global_attn_every_n_layers,
            global_rope_theta: c.rope_parameters.full_attention.rope_theta,
            local_attention: c.local_attention,
            local_rope_theta: c.rope_parameters.sliding_attention.rope_theta,
            classifier_config: None,
        }
    }
}

/// `rl_agent_config.json`: sequence budgets and calibration temperatures.
#[derive(serde::Deserialize)]
pub struct RlAgentConfig {
    pub max_len: usize,
    pub head_max_len: usize,
    #[serde(default)]
    pub temperature: Vec<f64>,
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f64>,
}

// ---------------------------------------------------------------------------
// Questions (Jev-compatible)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

/// One typed question. `options` are rendered exactly like the reference:
/// choice options are `key` or `key: description`, score options are
/// `level i: criterion`, noul options are `false: …` / `true: …`.
pub struct Question {
    pub qtype: QType,
    pub instructions: String,
    pub options: Vec<(String, String)>,
}

impl Question {
    pub fn choice(instructions: &str, criteria: &[(&str, &str)]) -> Self {
        Self {
            qtype: QType::Choice,
            instructions: instructions.to_string(),
            options: criteria
                .iter()
                .map(|(k, d)| (k.to_string(), d.to_string()))
                .collect(),
        }
    }

    pub fn score(instructions: &str, criteria: &[&str]) -> Self {
        Self {
            qtype: QType::Score,
            instructions: instructions.to_string(),
            options: criteria.iter().map(|c| (String::new(), c.to_string())).collect(),
        }
    }

    pub fn noul(instructions: &str) -> Self {
        Self {
            qtype: QType::Noul,
            instructions: instructions.to_string(),
            options: vec![
                ("false".into(), "no, the statement does not hold".into()),
                ("true".into(), "yes, the statement holds".into()),
            ],
        }
    }

    /// Option texts in label-index order (mirrors `render_options`).
    fn rendered_options(&self) -> Vec<String> {
        match self.qtype {
            QType::Choice => self
                .options
                .iter()
                .map(|(k, d)| if d.is_empty() { k.clone() } else { format!("{k}: {d}") })
                .collect(),
            QType::Score => self
                .options
                .iter()
                .enumerate()
                .map(|(i, (_, c))| format!("level {i}: {c}"))
                .collect(),
            QType::Noul => self
                .options
                .iter()
                .map(|(k, d)| format!("{k}: {d}"))
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Decision head
// ---------------------------------------------------------------------------

/// One `nn.TransformerEncoderLayer` (norm_first, relu FFN, eval-mode).
struct HeadLayer {
    in_proj_w: Tensor, // [3d, d]
    in_proj_b: Tensor, // [3d]
    out_proj: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    nhead: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, d: usize, nhead: usize) -> Result<Self> {
        Ok(Self {
            in_proj_w: vb.pp("self_attn").get((3 * d, d), "in_proj_weight")?,
            in_proj_b: vb.pp("self_attn").get(3 * d, "in_proj_bias")?,
            out_proj: linear(d, d, vb.pp("self_attn").pp("out_proj"))?,
            norm1: layer_norm(d, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(d, 1e-5, vb.pp("norm2"))?,
            linear1: linear(d, 4 * d, vb.pp("linear1"))?,
            linear2: linear(4 * d, d, vb.pp("linear2"))?,
            nhead,
        })
    }

    /// x: [B, L, d]; key_pad: additive [B, 1, 1, L] (0 real, large-negative pad).
    fn forward(&self, x: &Tensor, key_pad: &Tensor) -> Result<Tensor> {
        let (b, l, d) = x.dims3()?;
        let dh = d / self.nhead;
        let h = x.apply(&self.norm1)?;
        let in_proj = Linear::new(self.in_proj_w.clone(), Some(self.in_proj_b.clone()));
        let qkv = h.apply(&in_proj)?;
        let split = |off: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(2, off * d, d)?
                .reshape((b, l, self.nhead, dh))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let scale = 1.0 / (dh as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;
        let att = att.broadcast_add(key_pad)?;
        let att = softmax_last_dim(&att)?;
        let y = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, d))?;
        let x = (x + self.out_proj.forward(&y)?)?;
        let h2 = x.apply(&self.norm2)?;
        let ff = self.linear2.forward(&self.linear1.forward(&h2)?.relu()?)?;
        Ok((x + ff)?)
    }
}

struct DecisionHead {
    type_emb: Embedding,
    layers: Vec<HeadLayer>,
    scorer_norm: LayerNorm,
    scorer1: Linear,
    scorer2: Linear,
    act1: Linear,
    act2: Linear,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A loaded Laya decision model (encoder + decision head + calibration).
pub struct Laya {
    tok: tokenizers::Tokenizer,
    encoder: ModernBert,
    head: DecisionHead,
    cfg: RlAgentConfig,
    device: Device,
    cls: u32,
    sep: u32,
    pad: u32,
    mask: u32,
}

/// One answered question.
#[derive(Debug, Clone)]
pub struct Answer {
    /// The chosen option key (choice questions).
    pub choice: Option<String>,
    /// (option label, calibrated probability) in label order.
    pub probabilities: Vec<(String, f32)>,
    /// Expected level (score questions), in `0..k-1`.
    pub score: Option<f32>,
    /// P(statement holds) (noul questions).
    pub noul: Option<f32>,
    /// 1 - normalized entropy of the answer distribution.
    pub confidence: f32,
    /// P("answer now" action) from the RL act head.
    pub act_probability: f32,
}

#[derive(Debug)]
pub struct Decision {
    pub answers: Vec<(String, Answer)>,
    pub input_tokens: usize,
}

impl Laya {
    /// Download a Laya checkpoint from the Hub and load it on `device`.
    /// Expects the `convaiinnovations/laya` repo layout (`tokenizer/`,
    /// `encoder/config.json`, `model.safetensors`, `rl_agent_config.json`).
    pub fn from_hub(repo: &str, device: Device) -> Result<Self> {
        let tok_path = HfTokenizer::download(repo, "tokenizer/tokenizer.json")?;
        let enc_cfg_path = HfTokenizer::download(repo, "encoder/config.json")?;
        let weights_path = HfTokenizer::download(repo, "model.safetensors")?;
        let rl_cfg_path = HfTokenizer::download(repo, "rl_agent_config.json")?;
        Self::from_files(&tok_path, &enc_cfg_path, &weights_path, &rl_cfg_path, device)
    }

    /// Load from local files (see [`Laya::from_hub`] for the layout).
    pub fn from_files(
        tokenizer_json: &Path,
        encoder_config: &Path,
        safetensors: &Path,
        rl_agent_config: &Path,
        device: Device,
    ) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("failed to load Laya tokenizer: {e}"))?;
        let enc_cfg: EncoderConfig =
            serde_json::from_reader(std::fs::File::open(encoder_config)?)
                .context("failed to parse encoder/config.json")?;
        let mb_cfg = MbConfig::from(enc_cfg);
        let cfg: RlAgentConfig =
            serde_json::from_reader(std::fs::File::open(rl_agent_config)?)
                .context("failed to parse rl_agent_config.json")?;

        // Laya's state dict names the encoder subtree `encoder.*`; candle's
        // ModernBert::load hardcodes `model.*`. Rename while loading, then
        // load from the root (candle prepends `model.` itself).
        let raw = candle_core::safetensors::load(safetensors, &Device::Cpu)?;
        let tensors: HashMap<String, Tensor> = raw
            .into_iter()
            .map(|(name, t)| {
                let name = match name.strip_prefix("encoder.") {
                    Some(rest) => format!("model.{rest}"),
                    None => name,
                };
                (name, t)
            })
            .collect();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let encoder = ModernBert::load(vb.clone(), &mb_cfg)
            .context("failed to load ModernBERT encoder")?;

        let d = mb_cfg.hidden_size;
        let nhead = (d / 64).max(1);
        let mut layers = Vec::new();
        for i in 0..2 {
            layers.push(HeadLayer::load(vb.pp("head").pp(format!("layers.{i}")), d, nhead)?);
        }
        let head = DecisionHead {
            type_emb: embedding(3, d, vb.pp("type_emb"))?,
            layers,
            scorer_norm: layer_norm(d, 1e-5, vb.pp("scorer").pp("0"))?,
            scorer1: linear(d, d, vb.pp("scorer").pp("1"))?,
            scorer2: linear(d, 1, vb.pp("scorer").pp("3"))?,
            act1: linear(d + 4, 256, vb.pp("act_head").pp("0"))?,
            act2: linear(256, 2, vb.pp("act_head").pp("2"))?,
        };

        // English checkpoint uses [CLS]/[SEP]/[PAD]/[MASK]; the multilingual
        // (mmBERT) checkpoint names them <bos>/<eos>/<pad>/<mask> (per its
        // tokenizer_config: cls=<bos>, sep=<eos>). Try English first.
        let (cls, sep, pad, mask) = {
            let id = |cands: &[&str]| -> Result<u32> {
                cands
                    .iter()
                    .find_map(|t| tok.token_to_id(t))
                    .with_context(|| format!("tokenizer is missing any of {cands:?}"))
            };
            (
                id(&["[CLS]", "<bos>"])?,
                id(&["[SEP]", "<eos>"])?,
                id(&["[PAD]", "<pad>"])?,
                id(&["[MASK]", "<mask>"])?,
            )
        };
        Ok(Self {
            tok,
            encoder,
            head,
            cfg,
            device,
            cls,
            sep,
            pad,
            mask,
        })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tok
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    /// `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 … [SEP] state [SEP]`
    /// Returns (ids, marker positions) — a faithful port of `build_sequence`.
    fn build_sequence(&self, state: &str, q: &Question) -> Result<(Vec<u32>, Vec<u32>)> {
        let opts = q.rendered_options();
        let mask_str = "[MASK]";
        let ins = q.instructions.replace(mask_str, " ");
        let mut head_ids = self.encode(&format!("{} question: {}", q.qtype.name(), ins));
        let mut opt_ids: Vec<Vec<u32>> = opts
            .iter()
            .map(|o| {
                let mut ids = vec![self.mask];
                let mut rest = self.encode(&format!(" {}", o.replace(mask_str, " ")));
                rest.truncate(48);
                ids.append(&mut rest);
                ids
            })
            .collect();
        let opt_budget = self.cfg.head_max_len as isize - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        if opt_budget < 16 {
            let per = ((self.cfg.head_max_len - 16) / opt_ids.len().max(1)).max(4);
            for o in opt_ids.iter_mut() {
                o.truncate(per);
            }
        }
        let opt_budget = self.cfg.head_max_len as isize - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        head_ids.truncate((opt_budget.max(8) as usize).min(head_ids.len()));

        let mut ids = vec![self.cls];
        ids.extend_from_slice(&head_ids);
        ids.push(self.sep);
        let mut markers = Vec::with_capacity(opt_ids.len());
        for o in &opt_ids {
            markers.push(ids.len() as u32);
            ids.extend_from_slice(o);
        }
        ids.push(self.sep);
        let room = self.cfg.max_len.saturating_sub(ids.len() + 1);
        let mut st = self.encode(&state.replace(mask_str, " "));
        st.truncate(room);
        ids.extend_from_slice(&st);
        ids.push(self.sep);
        ids.truncate(self.cfg.max_len);
        markers.retain(|&m| (m as usize) < self.cfg.max_len);
        if markers.len() != opts.len() {
            anyhow::bail!(
                "question options do not fit in head_max_len={} ({} of {} markers kept)",
                self.cfg.head_max_len,
                markers.len(),
                opts.len()
            );
        }
        Ok((ids, markers))
    }

    fn temperature(&self, qtype: QType, k: usize) -> f64 {
        let size = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let bucket = format!("{}:{}", qtype.name(), size);
        self.cfg
            .temperature_by_options
            .get(&bucket)
            .copied()
            .or_else(|| self.cfg.temperature.get(qtype.index()).copied())
            .unwrap_or(1.0)
    }

    /// Run one batched decision pass: `questions` are `(id, question)` pairs
    /// over the shared `state`. Mirrors `RLAgent.system_one`.
    pub fn decide(&self, state: &str, questions: &[(String, Question)]) -> Result<Decision> {
        anyhow::ensure!(!questions.is_empty(), "no questions");
        let mut seqs = Vec::with_capacity(questions.len());
        for (_, q) in questions {
            seqs.push(self.build_sequence(state, q)?);
        }
        let b = seqs.len();
        let lmax = seqs.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);

        // Padded batch tensors.
        let mut ids_flat = vec![self.pad; b * lmax];
        let mut att_flat = vec![0f32; b * lmax];
        for (r, (ids, _)) in seqs.iter().enumerate() {
            ids_flat[r * lmax..r * lmax + ids.len()].copy_from_slice(ids);
            for a in att_flat[r * lmax..r * lmax + ids.len()].iter_mut() {
                *a = 1.0;
            }
        }
        let input_ids = Tensor::from_vec(ids_flat, (b, lmax), &self.device)?;
        let att = Tensor::from_vec(att_flat, (b, lmax), &self.device)?;
        let input_tokens: usize = seqs.iter().map(|(ids, _)| ids.len()).sum();

        // Encoder + per-row question-type embedding.
        let mut h = self.encoder.forward(&input_ids, &att)?;
        let qtypes = Tensor::from_vec(
            questions
                .iter()
                .map(|(_, q)| q.qtype.index() as u32)
                .collect::<Vec<_>>(),
            (b,),
            &self.device,
        )?;
        let temb = self.head.type_emb.forward(&qtypes)?.unsqueeze(1)?;
        h = h.broadcast_add(&temb)?;

        // Decision head (key padding mask -> additive bias).
        let key_pad = att.affine(-1.0, 1.0)?.affine(-1e9, 0.0)?.reshape((b, 1, 1, lmax))?;
        for layer in &self.head.layers {
            h = layer.forward(&h, &key_pad)?;
        }

        // Per-question: gather marker rows, score, calibrate, format.
        let mut answers = Vec::with_capacity(questions.len());
        for (r, (qid, q)) in questions.iter().enumerate() {
            let (_, markers) = &seqs[r];
            let k = markers.len();
            let pos = Tensor::from_vec(markers.clone(), (k,), &self.device)?;
            let m = h.get(r)?.index_select(&pos, 0)?; // [k, d]
            let logits = m
                .apply(&self.head.scorer_norm)?
                .apply(&self.head.scorer1)?
                .gelu()?
                .apply(&self.head.scorer2)?
                .squeeze(1)?; // [k]
            let raw: Vec<f32> = logits.to_vec1()?;

            // Calibrated answer probabilities.
            let temp = self.temperature(q.qtype, k);
            let zmax = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f32> = raw
                .iter()
                .map(|z| ((z - zmax) as f64 / temp).exp() as f32)
                .collect();
            let psum: f32 = p.iter().sum();
            for v in p.iter_mut() {
                *v /= psum;
            }

            // Act-head features from the UNCALIBRATED distribution (reference
            // behaviour: features are computed pre-temperature).
            let uzmax = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let up: Vec<f32> = raw.iter().map(|z| (z - uzmax).exp()).collect();
            let upsum: f32 = up.iter().sum();
            let up: Vec<f32> = up.iter().map(|v| v / upsum).collect();
            let ent: f32 = -up
                .iter()
                .map(|v| v * v.max(1e-9).ln())
                .sum::<f32>()
                / (k.max(2) as f32).ln();
            let top1 = up.iter().cloned().fold(0f32, f32::max);
            let top2 = {
                let mut s = up.clone();
                s.sort_by(|a, b| b.partial_cmp(a).unwrap());
                if k >= 2 { s[0] - s[1] } else { s[0] }
            };
            let pooled = h.get(r)?.get(0)?; // [d]
            let feats = Tensor::from_vec(vec![top1, top2, ent, k as f32 / 255.0], (4,), &self.device)?;
            let act_in = Tensor::cat(&[pooled, feats], 0)?.unsqueeze(0)?;
            let act_logits = self
                .head
                .act2
                .forward(&self.head.act1.forward(&act_in)?.gelu()?)?;
            let act_probs = softmax_last_dim(&act_logits)?.squeeze(0)?.to_vec1()?;

            let confidence = if k < 2 {
                1.0
            } else {
                let e: f32 = -p.iter().map(|v| v * v.clamp(1e-12, 1.0).ln()).sum::<f32>();
                1.0 - e / (k as f32).ln()
            };
            let options = q.rendered_options();
            let answer = match q.qtype {
                QType::Choice => {
                    let best = p
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    Answer {
                        choice: Some(q.options[best].0.clone()),
                        probabilities: q
                            .options
                            .iter()
                            .map(|(key, _)| key.clone())
                            .zip(p.iter().copied())
                            .collect(),
                        score: None,
                        noul: None,
                        confidence,
                        act_probability: act_probs[0],
                    }
                }
                QType::Score => {
                    let expected: f32 = p.iter().enumerate().map(|(i, v)| i as f32 * v).sum();
                    Answer {
                        choice: None,
                        probabilities: (0..k)
                            .map(|i| (i.to_string(), p[i]))
                            .collect(),
                        score: Some(expected),
                        noul: None,
                        confidence,
                        act_probability: act_probs[0],
                    }
                }
                QType::Noul => Answer {
                    choice: None,
                    probabilities: options.into_iter().zip(p.iter().copied()).collect(),
                    score: None,
                    noul: Some(p[1]),
                    confidence,
                    act_probability: act_probs[0],
                },
            };
            answers.push((qid.clone(), answer));
        }
        Ok(Decision {
            answers,
            input_tokens,
        })
    }
}
