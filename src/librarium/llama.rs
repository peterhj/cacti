use crate::prelude::*;
use crate::prelude_op::*;
use crate::librarium::lm::*;
use crate::util::cell::*;
use crate::util::pickle::*;

// This implementation of the llama architecture is derived
// (with our own simplifications/modifications/optimizations)
// from the model code by Eleuther/HuggingFace:

/* Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.

This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
and OPT implementations in this library. It has been modified from its
original forms to accommodate minor architectural differences compared
to GPT-NeoX and OPT used by the Meta AI team that trained the model.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// The hyperparameters for OpenLLaMa are derived from the
// original model code by Xinyang Geng and Hao Liu, and is
// distributed according to the Apache 2.0 license.

#[derive(Clone, Copy, Debug)]
pub struct LlamaConfig {
  pub num_layer: i64,
  pub tok_dim: i64,
  pub head_dim: i64,
  pub num_head: i64,
  pub mlp_inner_dim: i64,
  pub seq_cap: i64,
  pub ubat_sz: i64,
  pub rms_norm_eps: f32,
  pub dtype: Dtype,
}

impl LlamaConfig {
  pub fn open_llama_3b() -> LlamaConfig {
    LlamaConfig{
      num_layer:  26,
      tok_dim:    32000,
      head_dim:   100,
      num_head:   32,
      mlp_inner_dim:  8640,
      seq_cap:    2048,
      ubat_sz:    1,
      rms_norm_eps:   1.0e-6,
      dtype:      f16::dtype_(),
    }
  }

  pub fn open_llama_7b() -> LlamaConfig {
    LlamaConfig{
      num_layer:  32,
      tok_dim:    32000,
      head_dim:   128,
      num_head:   32,
      mlp_inner_dim:  11008,
      seq_cap:    2048,
      ubat_sz:    1,
      rms_norm_eps:   1.0e-6,
      dtype:      f16::dtype_(),
    }
  }

  pub fn open_llama_13b() -> LlamaConfig {
    LlamaConfig{
      num_layer:  40,
      tok_dim:    32000,
      head_dim:   128,
      num_head:   40,
      mlp_inner_dim:  13824,
      seq_cap:    2048,
      ubat_sz:    1,
      rms_norm_eps:   1.0e-6,
      dtype:      f16::dtype_(),
    }
  }
}

#[derive(Clone, Debug)]
pub struct LlamaLayer {
  pub pre_norm: StableCell,
  pub inv_freq: StableCell,
  pub q: StableCell,
  pub k: StableCell,
  pub v: StableCell,
  pub o: StableCell,
  pub post_norm: StableCell,
  pub gate: StableCell,
  pub up: StableCell,
  pub down: StableCell,
}

#[derive(Clone, Debug)]
pub struct Llama {
  pub cfg: LlamaConfig,
  pub cos: Option<StableCell>,
  pub sin: Option<StableCell>,
  pub embed: StableCell,
  pub layers: Vec<LlamaLayer>,
  pub head_norm: StableCell,
  pub lm_head: StableCell,
}

impl From<LlamaConfig> for Llama {
  fn from(cfg: LlamaConfig) -> Llama {
    let ubat_sz = cfg.ubat_sz;
    let seq_cap = cfg.seq_cap;
    let num_head = cfg.num_head;
    let head_dim = cfg.head_dim;
    let inner_dim = num_head * head_dim;
    let mlp_inner_dim = cfg.mlp_inner_dim;
    let tok_dim = cfg.tok_dim;
    let num_layers = cfg.num_layer as usize;
    assert!(tok_dim <= u16::max_value() as i64 + 1);
    let cos = None;
    let sin = None;
    let embed = StableCell::array([tok_dim, inner_dim], f16::dtype_());
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0 .. num_layers {
      let q = StableCell::array([inner_dim, inner_dim], f16::dtype_());
      let k = StableCell::array([inner_dim, inner_dim], f16::dtype_());
      let v = StableCell::array([inner_dim, inner_dim], f16::dtype_());
      let o = StableCell::array([inner_dim, inner_dim], f16::dtype_());
      let gate = StableCell::array([mlp_inner_dim, inner_dim], f16::dtype_());
      let up = StableCell::array([mlp_inner_dim, inner_dim], f16::dtype_());
      let down = StableCell::array([inner_dim, mlp_inner_dim], f16::dtype_());
      let inv_freq = StableCell::array([head_dim / 2], f32::dtype_());
      let pre_norm = StableCell::array([inner_dim], f16::dtype_());
      let post_norm = StableCell::array([inner_dim], f16::dtype_());
      layers.push(LlamaLayer{q, k, v, o, gate, down, up, inv_freq, pre_norm, post_norm});
    }
    let head_norm = StableCell::array([inner_dim], f16::dtype_());
    let lm_head = StableCell::array([tok_dim, inner_dim], f16::dtype_());
    Llama{cfg, cos, sin, embed, layers, head_norm, lm_head}
  }
}

impl Llama {
  pub fn match_pickle_dir(&self, pickdir: &PickleDir) -> CellInvMatches {
    let mut matcher = CellMatcher::new();
    matcher.insert("embed", self.embed.clone());
    for i in 0 .. self.layers.len() {
      matcher.insert((i, "attn", "q_proj"), self.layers[i].q.clone());
      matcher.insert((i, "attn", "k_proj"), self.layers[i].k.clone());
      matcher.insert((i, "attn", "v_proj"), self.layers[i].v.clone());
      matcher.insert((i, "attn", "o_proj"), self.layers[i].o.clone());
      matcher.insert((i, "mlp", "gate"), self.layers[i].gate.clone());
      matcher.insert((i, "mlp", "up"), self.layers[i].up.clone());
      matcher.insert((i, "mlp", "down"), self.layers[i].down.clone());
      matcher.insert((i, "inv_freq"), self.layers[i].inv_freq.clone());
      matcher.insert((i, "input_layernorm"), self.layers[i].pre_norm.clone());
      matcher.insert((i, "post_attention_layernorm"), self.layers[i].post_norm.clone());
    }
    matcher.insert("norm", self.head_norm.clone());
    matcher.insert("lm_head", self.lm_head.clone());
    let matches = matcher.match_(pickdir.clone_keys());
    matches.inv()
  }

  pub fn clone_param(&self) -> Vec<StableCell> {
    let mut param = Vec::new();
    param.push(self.embed.clone());
    for layer in self.layers.iter() {
      param.push(layer.pre_norm.clone());
      param.push(layer.q.clone());
      param.push(layer.k.clone());
      param.push(layer.v.clone());
      param.push(layer.o.clone());
      param.push(layer.post_norm.clone());
      param.push(layer.gate.clone());
      param.push(layer.up.clone());
      param.push(layer.down.clone());
    }
    param.push(self.head_norm.clone());
    param.push(self.lm_head.clone());
    param
  }

  /*pub fn fresh_grad(&self) -> Vec<StableCell> {
    // FIXME
    unimplemented!();
  }

  pub fn fresh_grad_with_matmul_dtype(&self, dtype: Dtype) -> Vec<StableCell> {
    // FIXME
    unimplemented!();
  }*/

  pub fn fresh_input(&self) -> LanguageModelIn {
    let ubat_sz = self.cfg.ubat_sz;
    let seq_cap = self.cfg.seq_cap;
    let in_tok = StableCell::array([ubat_sz, seq_cap], u16::dtype_());
    let in_lm_tok = StableCell::array([ubat_sz, seq_cap], u16::dtype_());
    let in_lm_loss_scale = StableCell::array([ubat_sz, seq_cap], f32::dtype_());
    LanguageModelIn{in_tok, in_lm_tok, in_lm_loss_scale}
  }

  pub fn init_constants(&mut self) {
    let init_embed = || {
      let seq_cap = self.cfg.seq_cap;
      let dtype = self.cfg.dtype;
      let pos = iota(seq_cap).cast(f32::dtype_());
      let freq = pos.outer_mul(&self.layers[0].inv_freq);
      let freq2 = freq.inner_concat(freq);
      let cos = freq2.cos().cast(dtype);
               //.new_shape([1, seq_cap, 1, head_dim])
               //.const_();
      let sin = freq2.sin().cast(dtype);
               //.new_shape([1, seq_cap, 1, head_dim])
               //.const_();
      (cos, sin)
    };
    let (cos, sin) = init_embed();
    self.cos = Some(cos.keep());
    self.sin = Some(sin.keep());
  }

  pub fn cache_constants(&self) {
    self.cos.as_ref().unwrap().cache();
    self.sin.as_ref().unwrap().cache();
  }

  pub fn apply<X: CellDeref, Y: CellDeref>(&self, in_tok: X, in_lm_tok: Y) -> LanguageModelOut {
    let ubat_sz = self.cfg.ubat_sz;
    let seq_cap = self.cfg.seq_cap;
    let num_head = self.cfg.num_head;
    let head_dim = self.cfg.head_dim;
    let inner_dim = num_head * head_dim;
    let mlp_inner_dim = self.cfg.mlp_inner_dim;
    let tok_dim = self.cfg.tok_dim;
    let num_layer = self.cfg.num_layer;
    let rms_norm_eps = self.cfg.rms_norm_eps;
    let in_tok = in_tok._deref().const_();
    let rms_norm = |x: &CellPtr, weight: &StableCell, eps: f32, dtype: Dtype| {
      let x = x.new_shape([ubat_sz * seq_cap, num_head * head_dim]);
      let v = x.cast(f32::dtype_())
               .square().inner_mean()
               .new_shape([ubat_sz * seq_cap, 1]);
      let t = x / (v + eps).sqrt();
      let w = weight.new_shape([1, num_head * head_dim]);
      let y = (w * t).cast(dtype);
      y
    };
    /*let inner_symplectic_map = |x: CellPtr| {
      let xty = x.type_();
      let (lx, rx) = x.inner_split(xty.inner_len() / 2);
      (-rx).inner_concat(lx)
    };*/
    let cos = self.cos.as_ref().unwrap()
             .new_shape([1, seq_cap, 1, head_dim])
             .const_();
    let sin = self.sin.as_ref().unwrap()
             .new_shape([1, seq_cap, 1, head_dim])
             .const_();
    let symplectic_embed = |x: CellPtr| {
      (x * cos) + (inner_symplectic_map(x) * sin)
    };
    let block_causal_attention_mask = |x: CellPtr| {
      x.block_tri_elem_affine(1.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, -f32::inf())
    };
    let mut stream = in_tok;
    stream = self.embed.outer_select(stream)
            .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
    for ell in 0 .. num_layer as usize {
      let pre_nrm = rms_norm(&stream, &self.layers[ell].pre_norm, rms_norm_eps, f16::dtype_());
      let q_proj = symplectic_embed(
                   pre_nrm
                  .matmul(false, &self.layers[ell].q, true)
                  .new_shape([ubat_sz, seq_cap, num_head, head_dim])
                   );
      let k_proj = symplectic_embed(
                   pre_nrm
                  .matmul(false, &self.layers[ell].k, true)
                  .new_shape([ubat_sz, seq_cap, num_head, head_dim])
                   );
      let v_proj = pre_nrm
                  .matmul(false, &self.layers[ell].v, true)
                  .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      let attn = q_proj.block_matmul_scale(false, k_proj, true, 1.0 / (head_dim as f32).sqrt())
                .cast(f32::dtype_());
      let attn = block_causal_attention_mask(attn)
                .inner_softmax()
                .cast(f16::dtype_());
      let v_attn = attn.block_matmul(false, v_proj, false)
                  .new_shape([ubat_sz * seq_cap, num_head * head_dim]);
      let o_proj = v_attn.matmul(false, &self.layers[ell].o, true)
                  .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      stream = stream + o_proj;
      let post_nrm = rms_norm(&stream, &self.layers[ell].post_norm, rms_norm_eps, f16::dtype_());
      let up_proj = post_nrm
                   .matmul(false, &self.layers[ell].up, true);
      let gate_proj = post_nrm
                     .matmul(false, &self.layers[ell].gate, true);
      let gate_proj = gate_proj.standard_silu();
      let gate_up = gate_proj * up_proj;
      let down_proj = gate_up.matmul(false, &self.layers[ell].down, true)
                     .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      stream = stream + down_proj;
    }
    let stream = rms_norm(&stream, &self.head_norm, rms_norm_eps, f16::dtype_());
    let out_lm_logit = stream
                      .matmul(false, &self.lm_head, true)
                      .new_shape([ubat_sz, seq_cap, tok_dim])
                      .keep();
    let logit32 = out_lm_logit.cast(f32::dtype_());
    let out_lm_prob = logit32.inner_softmax().keep();
    let in_lm_tok = in_lm_tok.const_();
    let out_lm_loss = logit32.inner_softmax_categorical_nll(in_lm_tok).keep();
    LanguageModelOut{out_lm_logit, out_lm_prob, out_lm_loss}
  }
}

#[derive(Clone, Debug)]
pub struct LlamaCachedState {
  pub k_cache: StableCell,
  pub v_cache: StableCell,
}

#[derive(Clone, Debug)]
pub struct LlamaCached {
  pub cfg: LlamaConfig,
  pub cos: Option<StableCell>,
  pub sin: Option<StableCell>,
  pub embed: StableCell,
  pub layers: Vec<LlamaLayer>,
  pub states: Vec<LlamaCachedState>,
  pub head_norm: StableCell,
  pub lm_head: StableCell,
}

impl From<LlamaConfig> for LlamaCached {
  fn from(cfg: LlamaConfig) -> LlamaCached {
    let ubat_sz = cfg.ubat_sz;
    let seq_cap = cfg.seq_cap;
    let num_head = cfg.num_head;
    let head_dim = cfg.head_dim;
    let inner_dim = num_head * head_dim;
    let mlp_inner_dim = cfg.mlp_inner_dim;
    let tok_dim = cfg.tok_dim;
    let num_layers = cfg.num_layer as usize;
    assert!(tok_dim <= u16::max_value() as i64 + 1);
    let cos = None;
    let sin = None;
    let embed = StableCell::array([tok_dim, inner_dim], f16::dtype_());
    let mut layers = Vec::with_capacity(num_layers);
    let mut states = Vec::with_capacity(num_layers);
    for _ in 0 .. num_layers {
      let q = StableCell::array([inner_dim, inner_dim], f16::dtype_());
      let k = StableCell::array([inner_dim, inner_dim], f16::dtype_());
      let v = StableCell::array([inner_dim, inner_dim], f16::dtype_());
      let o = StableCell::array([inner_dim, inner_dim], f16::dtype_());
      let gate = StableCell::array([mlp_inner_dim, inner_dim], f16::dtype_());
      let up = StableCell::array([mlp_inner_dim, inner_dim], f16::dtype_());
      let down = StableCell::array([inner_dim, mlp_inner_dim], f16::dtype_());
      let inv_freq = StableCell::array([head_dim / 2], f32::dtype_());
      let pre_norm = StableCell::array([inner_dim], f16::dtype_());
      let post_norm = StableCell::array([inner_dim], f16::dtype_());
      layers.push(LlamaLayer{q, k, v, o, gate, down, up, inv_freq, pre_norm, post_norm});
      let k_cache = StableCell::array([ubat_sz, seq_cap, num_head, head_dim], f16::dtype_());
      let v_cache = StableCell::array([ubat_sz, seq_cap, num_head, head_dim], f16::dtype_());
      states.push(LlamaCachedState{k_cache, v_cache});
    }
    let head_norm = StableCell::array([inner_dim], f16::dtype_());
    let lm_head = StableCell::array([tok_dim, inner_dim], f16::dtype_());
    LlamaCached{cfg, cos, sin, embed, layers, states, head_norm, lm_head}
  }
}

impl LlamaCached {
  pub fn match_pickle_dir(&self, pickdir: &PickleDir) -> CellInvMatches {
    let mut matcher = CellMatcher::new();
    matcher.insert("embed", self.embed.clone());
    for i in 0 .. self.layers.len() {
      matcher.insert((i, "attn", "q_proj"), self.layers[i].q.clone());
      matcher.insert((i, "attn", "k_proj"), self.layers[i].k.clone());
      matcher.insert((i, "attn", "v_proj"), self.layers[i].v.clone());
      matcher.insert((i, "attn", "o_proj"), self.layers[i].o.clone());
      matcher.insert((i, "mlp", "gate"), self.layers[i].gate.clone());
      matcher.insert((i, "mlp", "up"), self.layers[i].up.clone());
      matcher.insert((i, "mlp", "down"), self.layers[i].down.clone());
      matcher.insert((i, "inv_freq"), self.layers[i].inv_freq.clone());
      matcher.insert((i, "input_layernorm"), self.layers[i].pre_norm.clone());
      matcher.insert((i, "post_attention_layernorm"), self.layers[i].post_norm.clone());
    }
    matcher.insert("norm", self.head_norm.clone());
    matcher.insert("lm_head", self.lm_head.clone());
    let matches = matcher.match_(pickdir.clone_keys());
    matches.inv()
  }

  pub fn clone_param(&self) -> Vec<StableCell> {
    let mut param = Vec::new();
    param.push(self.embed.clone());
    for layer in self.layers.iter() {
      param.push(layer.pre_norm.clone());
      param.push(layer.q.clone());
      param.push(layer.k.clone());
      param.push(layer.v.clone());
      param.push(layer.o.clone());
      param.push(layer.post_norm.clone());
      param.push(layer.gate.clone());
      param.push(layer.up.clone());
      param.push(layer.down.clone());
    }
    param.push(self.head_norm.clone());
    param.push(self.lm_head.clone());
    param
  }

  pub fn fresh_input(&self) -> Vec<LanguageModelDeployIn> {
    let ubat_sz = self.cfg.ubat_sz;
    let seq_cap = self.cfg.seq_cap;
    let mut in_ = Vec::with_capacity(ubat_sz as usize);
    for _ in 0 .. ubat_sz {
      let in_tok = StableCell::array([1, seq_cap], u16::dtype_());
      in_.push(LanguageModelDeployIn{in_tok});
    }
    in_
  }

  pub fn init_constants(&mut self) {
    let init_embed = || {
      let seq_cap = self.cfg.seq_cap;
      let dtype = self.cfg.dtype;
      let pos = iota(seq_cap).cast(f32::dtype_());
      let freq = pos.outer_mul(&self.layers[0].inv_freq);
      let freq2 = freq.inner_concat(freq);
      let cos = freq2.cos().cast(dtype);
               //.new_shape([1, seq_cap, 1, head_dim])
               //.const_();
      let sin = freq2.sin().cast(dtype);
               //.new_shape([1, seq_cap, 1, head_dim])
               //.const_();
      (cos, sin)
    };
    let (cos, sin) = init_embed();
    self.cos = Some(cos.keep());
    self.sin = Some(sin.keep());
  }

  pub fn cache_constants(&self) {
    self.cos.as_ref().unwrap().cache();
    self.sin.as_ref().unwrap().cache();
  }

  pub fn init_state(&mut self) {
    let ubat_sz = self.cfg.ubat_sz;
    let seq_cap = self.cfg.seq_cap;
    let num_head = self.cfg.num_head;
    let head_dim = self.cfg.head_dim;
    let dtype = self.cfg.dtype;
    self.states.clear();
    for _ in 0 .. self.cfg.num_layer as usize {
      let k_cache = zeros([ubat_sz, seq_cap, num_head, head_dim], dtype).keep();
      let v_cache = zeros([ubat_sz, seq_cap, num_head, head_dim], dtype).keep();
      self.states.push(LlamaCachedState{
        k_cache,
        v_cache,
      });
    }
  }

  pub fn reset_state(&self) {
    for ell in 0 .. self.cfg.num_layer as usize {
      self.states[ell].k_cache.set_zeros();
      self.states[ell].v_cache.set_zeros();
    }
  }

  pub fn cache_state(&self) {
    for ell in 0 .. self.cfg.num_layer as usize {
      self.states[ell].k_cache.cache();
      self.states[ell].v_cache.cache();
    }
  }

  /*pub fn apply<X: CellDeref>(&self, in_tok: X) -> LanguageModelDeployOut {
    unimplemented!();
  }*/

  /*pub fn apply<X: CellDeref>(&mut self, in_tok: &mut [X], prev_seq_len: i64, next_seq_len: i64) -> LanguageModelBatchDeployOut {
    unimplemented!();
  }*/

  pub fn apply(&mut self, in_: &mut [LanguageModelDeployIn], prev_seq_len: i64, next_seq_len: i64) -> Vec<LanguageModelDeployOut> {
    let ubat_sz = self.cfg.ubat_sz;
    let seq_cap = self.cfg.seq_cap;
    let num_head = self.cfg.num_head;
    let head_dim = self.cfg.head_dim;
    let inner_dim = num_head * head_dim;
    let mlp_inner_dim = self.cfg.mlp_inner_dim;
    let tok_dim = self.cfg.tok_dim;
    let num_layer = self.cfg.num_layer;
    let rms_norm_eps = self.cfg.rms_norm_eps;
    let diff_seq_len = next_seq_len - prev_seq_len;
    assert_eq!(ubat_sz as usize, in_.len());
    assert!(diff_seq_len > 0);
    let rms_norm = |x: &CellPtr, weight: &StableCell, eps: f32, dtype: Dtype| {
      let x = x.new_shape([1 * diff_seq_len, num_head * head_dim]);
      let v = x.cast(f32::dtype_())
               .square().inner_mean()
               .new_shape([1 * diff_seq_len, 1]);
      let t = x / (v + eps).sqrt();
      let w = weight.new_shape([1, num_head * head_dim]);
      let y = (w * t).cast(dtype);
      y
    };
    /*let inner_symplectic_map = |x: CellPtr| {
      let xty = x.type_();
      let (lx, rx) = x.inner_split(xty.inner_len() / 2);
      (-rx).inner_concat(lx)
    };*/
    let cos = self.cos.as_ref().unwrap().clone()
             .new_shape([1, seq_cap, 1, head_dim])
              [(.., prev_seq_len .. next_seq_len, .., ..)].const_();
    let sin = self.sin.as_ref().unwrap().clone()
             .new_shape([1, seq_cap, 1, head_dim])
              [(.., prev_seq_len .. next_seq_len, .., ..)].const_();
    let symplectic_embed = |x: CellPtr| {
      (x * cos) + (inner_symplectic_map(x) * sin)
    };
    let mut stream = Vec::with_capacity(ubat_sz as usize);
    for i in 0 .. ubat_sz {
      let in_tok = &in_[i as usize].in_tok._deref()[(.., prev_seq_len .. next_seq_len)];
      stream.push(self.embed.outer_select(in_tok)
                 .new_shape([1, diff_seq_len, num_head, head_dim]));
    }
    for ell in 0 .. num_layer as usize {
      for i in 0 .. ubat_sz {
        let idx = i as usize;
        let pre_nrm =
               rms_norm(&stream[idx], &self.layers[ell].pre_norm, rms_norm_eps, f16::dtype_())
              .new_shape([1, diff_seq_len, num_head, head_dim]);
        let q_proj =
               symplectic_embed(
               pre_nrm
              .new_shape([diff_seq_len, num_head * head_dim])
              .matmul(false, &self.layers[ell].q, true)
              .new_shape([1, diff_seq_len, num_head, head_dim])
               );
        self.states[ell].k_cache[(i .. i + 1, prev_seq_len .. next_seq_len, .., ..)]
            += symplectic_embed(
               pre_nrm
              .new_shape([diff_seq_len, num_head * head_dim])
              .matmul(false, &self.layers[ell].k, true)
              .new_shape([1, diff_seq_len, num_head, head_dim])
               );
        self.states[ell].v_cache[(i .. i + 1, prev_seq_len .. next_seq_len, .., ..)]
            += pre_nrm
              .new_shape([diff_seq_len, num_head * head_dim])
              .matmul(false, &self.layers[ell].v, true)
              .new_shape([1, diff_seq_len, num_head, head_dim]);
        let attn =
               q_proj
              .block_matmul_scale(
                   false,
                   &self.states[ell].k_cache[(i .. i + 1, .. /*next_seq_len*/, .., ..)],
                   true,
                   1.0 / (head_dim as f32).sqrt()
               )
              .cast(f32::dtype_());
        let attn = row_causal_attention_mask(attn, prev_seq_len)
                  .inner_softmax()
                  .cast(f16::dtype_());
        let v_attn =
               attn
              .block_matmul(
                   false,
                   &self.states[ell].v_cache[(i .. i + 1, .. /*next_seq_len*/, .., ..)],
                   false
               )
              .new_shape([diff_seq_len, num_head * head_dim]);
        let o_proj = v_attn.matmul(false, &self.layers[ell].o, true)
                    .new_shape([1, diff_seq_len, num_head, head_dim]);
        stream[idx] = stream[idx] + o_proj;
        let post_nrm = rms_norm(&stream[idx], &self.layers[ell].post_norm, rms_norm_eps, f16::dtype_());
        let up_proj = post_nrm
                     .matmul(false, &self.layers[ell].up, true);
        let gate_proj = post_nrm
                       .matmul(false, &self.layers[ell].gate, true);
        let gate_proj = gate_proj.standard_silu();
        let gate_up = gate_proj * up_proj;
        let down_proj = gate_up.matmul(false, &self.layers[ell].down, true)
                       .new_shape([1, diff_seq_len, num_head, head_dim]);
        stream[idx] = stream[idx] + down_proj;
      }
    }
    let mut out = Vec::with_capacity(ubat_sz as usize);
    for i in 0 .. ubat_sz {
      let idx = i as usize;
      stream[idx] = rms_norm(&stream[idx], &self.head_norm, rms_norm_eps, f16::dtype_());
      let out_lm_logit = stream[idx]
                        .matmul(false, &self.lm_head, true)
                        .new_shape([1, diff_seq_len, tok_dim])
                        .keep();
      let out_lm_prob = out_lm_logit
                       .cast(f32::dtype_())
                       .inner_softmax()
                       .keep();
      let out_lm_tok = out_lm_logit
                      .inner_arg_max()
                      .cast(u16::dtype_())
                      .keep();
      in_[idx].in_tok._deref()[(.., next_seq_len .. next_seq_len + 1)]
          += &out_lm_tok[(.., diff_seq_len - 1 .. diff_seq_len)];
      out.push(LanguageModelDeployOut{out_lm_logit, out_lm_prob, out_lm_tok});
    }
    out
  }
}

#[track_caller]
pub fn inner_symplectic_map<X: CellDeref>(x: X) -> CellPtr {
  panick_wrap(|| {
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(x._deref());
    ctx_pop_thunk(InnerSymplecticMapFutThunkSpec)
  })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerSymplecticMapFutThunkSpec;

impl FutharkThunkSpec for InnerSymplecticMapFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_symplectic_map")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match out[0].ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_suf = {{%0.s[0]}} // 2 in"));
        code.append(format!(r"let (tl, tr) = split ({{%0}} :> [a_suf + a_suf]{}) in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let {{%1}} = ((map (\u -> (-u)) tr) ++ tl) :> [{{%0.s[0]}}]{} in",
            arg[0].dtype.format_futhark(),
        ));
      }
      2 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_suf = {{%0.s[1]}} // 2 in"));
        code.append(format!(r"let {{%1}} = map (\t -> let (tl, tr) = split (t :> [a_suf + a_suf]{}) in ((map (\u -> (-u)) tr) ++ tl) :> [{{%0.s[1]}}]{}) {{%0}} in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
      }
      3 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_suf = {{%0.s[2]}} // 2 in"));
        code.append(format!(r"let {{%1}} = map (\t1 -> map (\t2 -> let (tl, tr) = split (t2 :> [a_suf + a_suf]{}) in ((map (\u -> (-u)) tr) ++ tl) :> [{{%0.s[2]}}]{}) t1) {{%0}} in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
      }
      4 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_suf = {{%0.s[3]}} // 2 in"));
        code.append(format!(r"let {{%1}} = map (\t1 -> map (\t2 -> map (\t3 -> let (tl, tr) = split (t3 :> [a_suf + a_suf]{}) in ((map (\u -> (-u)) tr) ++ tl) :> [{{%0.s[3]}}]{}) t2) t1) {{%0}} in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }
}

// TODO

#[track_caller]
pub fn row_causal_attention_mask<X: CellDeref>(x: X, initial_row: i64) -> CellPtr {
  panick_wrap(|| {
    assert!(ctx_clean_arg());
    ctx_push_scalar_param(initial_row);
    ctx_push_cell_arg(x._deref());
    ctx_pop_thunk(RowCausalAttentionMaskFutThunkSpec)
  })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RowCausalAttentionMaskFutThunkSpec;

impl FutharkThunkSpec for RowCausalAttentionMaskFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.row_causal_attention_mask")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  fn param_count(&self) -> u16 {
    1
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.param_ct = 1;
    code.abi.set_param(0, FutharkParam::Spec, FutAbiScalarType::I64);
    match out[0].ndim {
      4 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let row_idxs = iota {{%0.s[1]}} in"));
        code.append(format!(r"let col_idxs = iota {{%0.s[3]}} in"));
        code.append(format!(r"let {{%1}} = map (\t1 -> map2 (\row_idx t2 -> map (\t3 -> map2 (\col_idx u -> if (row_idx + {{%param[0]}}) >= col_idx then u else (-{}.inf)) col_idxs t3) t2) row_idxs t1) {{%0}} in",
            arg[0].dtype.format_futhark(),
        ));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }
}
