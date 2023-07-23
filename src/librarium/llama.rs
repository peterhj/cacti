use crate::prelude::*;
use crate::librarium::lm::*;
use crate::util::cell::*;
use crate::util::pickle::*;

use std::borrow::{Borrow};

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
      dtype:      f16::dtype(),
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
      dtype:      f16::dtype(),
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
      dtype:      f16::dtype(),
    }
  }
}

#[derive(Clone)]
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

#[derive(Clone)]
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
    //let cos = StableCell::array([seq_cap, head_dim], f16::dtype());
    //let sin = StableCell::array([seq_cap, head_dim], f16::dtype());
    let cos = None;
    let sin = None;
    let embed = StableCell::array([tok_dim, inner_dim], f16::dtype());
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0 .. num_layers {
      let q = StableCell::array([1, inner_dim, 1, inner_dim], f16::dtype());
      let k = StableCell::array([1, inner_dim, 1, inner_dim], f16::dtype());
      let v = StableCell::array([1, inner_dim, 1, inner_dim], f16::dtype());
      let o = StableCell::array([1, inner_dim, 1, inner_dim], f16::dtype());
      let gate = StableCell::array([1, mlp_inner_dim, 1, inner_dim], f16::dtype());
      let up = StableCell::array([1, mlp_inner_dim, 1, inner_dim], f16::dtype());
      let down = StableCell::array([1, inner_dim, 1, mlp_inner_dim], f16::dtype());
      let inv_freq = StableCell::array([head_dim / 2], f32::dtype());
      let pre_norm = StableCell::array([inner_dim], f16::dtype());
      let post_norm = StableCell::array([inner_dim], f16::dtype());
      layers.push(LlamaLayer{q, k, v, o, gate, down, up, inv_freq, pre_norm, post_norm});
    }
    let head_norm = StableCell::array([inner_dim], f16::dtype());
    let lm_head = StableCell::array([1, tok_dim, 1, inner_dim], f16::dtype());
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
    // FIXME: 0-th embed requires zero grad.
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

  pub fn param(&self) -> Vec<StableCell> {
    self.clone_param()
  }

  pub fn fresh_grad(&self) -> Vec<StableCell> {
    // FIXME
    unimplemented!();
  }

  pub fn fresh_grad_with_matmul_dtype(&self, dtype: Dtype) -> Vec<StableCell> {
    // FIXME
    unimplemented!();
  }

  pub fn fresh_input(&self) -> LanguageModelIn {
    let ubat_sz = self.cfg.ubat_sz;
    let seq_cap = self.cfg.seq_cap;
    let in_tok = StableCell::array([ubat_sz, seq_cap], u16::dtype());
    let in_lm_tok = StableCell::array([ubat_sz, seq_cap], u16::dtype());
    LanguageModelIn{in_tok, in_lm_tok}
  }

  pub fn make_input(&self) -> LanguageModelIn {
    self.fresh_input()
  }

  pub fn init(&mut self) {
    // TODO
    let init_embed = || {
      let seq_cap = self.cfg.seq_cap;
      let dtype = self.cfg.dtype;
      let pos = iota(seq_cap).cast(f32::dtype());
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

  pub fn cache(&self) {
    // TODO
    self.cos.as_ref().unwrap().cache();
    self.sin.as_ref().unwrap().cache();
  }

  pub fn apply<X: Borrow<CellPtr>, Y: Borrow<CellPtr>>(&self, in_tok: X, in_lm_tok: Y) -> LanguageModelOut {
    let block_causal_attention_mask = |x: CellPtr| {
      x.block_tri_elem_affine(1.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, -f32::inf())
    };
    let rms_norm = |x: &CellPtr, weight: &StableCell, eps: f32, dtype: Dtype| {
      let ubat_sz = self.cfg.ubat_sz;
      let seq_cap = self.cfg.seq_cap;
      let num_head = self.cfg.num_head;
      let head_dim = self.cfg.head_dim;
      let x = x.new_shape([ubat_sz * seq_cap, num_head * head_dim]);
      let v = x.cast(f32::dtype())
               .square().inner_mean()
               .new_shape([ubat_sz * seq_cap, 1]);
      let t = x / (v + eps).sqrt();
      let w = weight.new_shape([1, num_head * head_dim]);
      let y = (w * t).cast(dtype)
                     .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      y
    };
    /*let inner_symplectic_map = |x: CellPtr| {
      let xty = x.type_();
      let (lx, rx) = x.inner_split(xty.inner_len() / 2);
      (-rx).inner_concat(lx)
    };*/
    let symplectic_embed = |q: CellPtr, k: CellPtr, | {
      let seq_cap = self.cfg.seq_cap;
      let head_dim = self.cfg.head_dim;
      let cos = self.cos.as_ref().unwrap()
               .new_shape([1, seq_cap, 1, head_dim])
               .const_();
      let sin = self.sin.as_ref().unwrap()
               .new_shape([1, seq_cap, 1, head_dim])
               .const_();
      let q = (q * cos) + (q.inner_symplectic_map() * sin);
      let k = (k * cos) + (k.inner_symplectic_map() * sin);
      (q, k)
    };
    let ubat_sz = self.cfg.ubat_sz;
    let seq_cap = self.cfg.seq_cap;
    let num_head = self.cfg.num_head;
    let head_dim = self.cfg.head_dim;
    let inner_dim = num_head * head_dim;
    let mlp_inner_dim = self.cfg.mlp_inner_dim;
    let tok_dim = self.cfg.tok_dim;
    let num_layer = self.cfg.num_layer;
    let rms_norm_eps = self.cfg.rms_norm_eps;
    let mut stream = *in_tok.borrow();
    /*stream = stream.inner_one_hot(tok_dim, f16::dtype());
    stream = stream
            .new_shape([ubat_sz * seq_cap, tok_dim])
            .block_mm([ubat_sz * seq_cap, tok_dim], false, &self.embed, [tok_dim, inner_dim], false)
            .new_shape([ubat_sz, seq_cap, num_head, head_dim]);*/
    stream = self.embed.outer_select(stream)
            .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
    for i in 0 .. num_layer as usize {
      let pre_nrm = rms_norm(&stream, &self.layers[i].pre_norm, rms_norm_eps, f16::dtype());
      let q_proj = pre_nrm
                  //.new_shape([ubat_sz * seq_cap, num_head * head_dim])
                  //.block_mm([ubat_sz * seq_cap, inner_dim], false, &self.layers[i].q, [inner_dim, inner_dim], true)
                  .new_shape([1, ubat_sz * seq_cap, 1, num_head * head_dim])
                  .block_matmul(false, &self.layers[i].q, true)
                  .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      let k_proj = pre_nrm
                  //.new_shape([ubat_sz * seq_cap, num_head * head_dim])
                  //.block_mm([ubat_sz * seq_cap, inner_dim], false, &self.layers[i].k, [inner_dim, inner_dim], true)
                  .new_shape([1, ubat_sz * seq_cap, 1, num_head * head_dim])
                  .block_matmul(false, &self.layers[i].k, true)
                  .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      let v_proj = pre_nrm
                  //.new_shape([ubat_sz * seq_cap, num_head * head_dim])
                  //.block_mm([ubat_sz * seq_cap, inner_dim], false, &self.layers[i].v, [inner_dim, inner_dim], true)
                  .new_shape([1, ubat_sz * seq_cap, 1, num_head * head_dim])
                  .block_matmul(false, &self.layers[i].v, true)
                  .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      let (q_proj, k_proj) = symplectic_embed(q_proj, k_proj);
      //let q_proj = q_proj.new_shape([ubat_sz * seq_cap, num_head * head_dim]);
      //let k_proj = k_proj.new_shape([ubat_sz * seq_cap, num_head * head_dim]);
      let attn = q_proj
                //.block_mm_scale([seq_cap, head_dim], false, k_proj, [seq_cap, head_dim], true, 1.0 / (head_dim as f32).sqrt())
                .block_matmul_scale(false, k_proj, true, 1.0 / (head_dim as f32).sqrt())
                //.new_shape([ubat_sz, seq_cap, num_head, seq_cap])
                .cast(f32::dtype());
      let attn = block_causal_attention_mask(attn)
                .inner_softmax()
                .cast(f16::dtype())
                //.new_shape([ubat_sz * seq_cap, num_head * seq_cap]);
                //.new_shape([ubat_sz, seq_cap, num_head, seq_cap]);
                ;
      /*// FIXME: pad because block_mm is a very low level wrapper.
      let pad_head_dim = ((head_dim + 8 - 1) / 8) * 8;
      let v_attn = if head_dim == pad_head_dim {
        let v_proj = v_proj.new_shape([ubat_sz * seq_cap, num_head * head_dim]);
        let v_attn = attn.block_mm([seq_cap, seq_cap], false, v_proj, [seq_cap, head_dim], false);
        v_attn
      } else {
        let v_proj = v_proj
                    .block_pad([seq_cap, pad_head_dim], f16::zero())
                    .new_shape([ubat_sz * seq_cap, num_head * pad_head_dim]);
        let v_attn = attn.block_mm([seq_cap, seq_cap], false, v_proj, [seq_cap, pad_head_dim], false)
                    .new_shape([ubat_sz, seq_cap, num_head, pad_head_dim])
                    .block_unpad([seq_cap, head_dim])
                    .new_shape([ubat_sz * seq_cap, num_head * head_dim]);
        v_attn
      };*/
      let v_attn = attn.block_matmul(false, v_proj, false)
                  //.new_shape([ubat_sz * seq_cap, num_head * head_dim]);
                  .new_shape([1, ubat_sz * seq_cap, 1, num_head * head_dim]);
      //let o_proj = v_attn.block_mm([ubat_sz * seq_cap, inner_dim], false, &self.layers[i].o, [inner_dim, inner_dim], true)
      let o_proj = v_attn.block_matmul(false, &self.layers[i].o, true)
                  .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      stream = stream + o_proj;
      let post_nrm = rms_norm(&stream, &self.layers[i].post_norm, rms_norm_eps, f16::dtype());
      let up_proj = post_nrm
                   //.new_shape([ubat_sz * seq_cap, num_head * head_dim])
                   //.block_mm([ubat_sz * seq_cap, inner_dim], false, &self.layers[i].up, [mlp_inner_dim, inner_dim], true);
                   .new_shape([1, ubat_sz * seq_cap, 1, num_head * head_dim])
                   .block_matmul(false, &self.layers[i].up, true);
      let gate_proj = post_nrm
                     //.new_shape([ubat_sz * seq_cap, num_head * head_dim])
                     //.block_mm([ubat_sz * seq_cap, inner_dim], false, &self.layers[i].gate, [mlp_inner_dim, inner_dim], true);
                     .new_shape([1, ubat_sz * seq_cap, 1, num_head * head_dim])
                     .block_matmul(false, &self.layers[i].gate, true);
      let gate_proj = gate_proj.standard_silu();
      let gate_up = gate_proj * up_proj;
      let down_proj = gate_up
                     //.block_mm([ubat_sz * seq_cap, mlp_inner_dim], false, &self.layers[i].down, [inner_dim, mlp_inner_dim], true)
                     .block_matmul(false, &self.layers[i].down, true)
                     .new_shape([ubat_sz, seq_cap, num_head, head_dim]);
      stream = stream + down_proj;
    }
    let stream = rms_norm(&stream, &self.head_norm, rms_norm_eps, f16::dtype());
    let out_lm_logit = stream
                      //.new_shape([ubat_sz * seq_cap, num_head * head_dim])
                      //.block_mm([ubat_sz * seq_cap, inner_dim], false, &self.lm_head, [tok_dim, inner_dim], true)
                      .new_shape([1, ubat_sz * seq_cap, 1, num_head * head_dim])
                      .block_matmul(false, &self.lm_head, true)
                      .new_shape([ubat_sz, seq_cap, tok_dim])
                      .keep();
    let logit32 = out_lm_logit.cast(f32::dtype());
    let out_lm_prob = logit32.inner_softmax().keep();
    /*let out_lm_loss = (-out_lm_prob.inner_select(in_lm_tok).ln()).keep();*/
    let out_lm_loss = logit32.inner_softmax_categorical_nll(in_lm_tok).keep();
    LanguageModelOut{out_lm_logit, out_lm_prob, out_lm_loss}
  }
}

#[derive(Clone)]
pub struct LlamaLayerCachedKV {
  // TODO
  pub q_cache: StableCell,
  pub k_cache: StableCell,
  pub v_cache: StableCell,
}

#[derive(Clone)]
pub struct LlamaCached {
  pub cfg: LlamaConfig,
  pub cos: Option<StableCell>,
  pub sin: Option<StableCell>,
  pub embed: StableCell,
  pub layers: Vec<(LlamaLayer, LlamaLayerCachedKV)>,
  pub head_norm: StableCell,
  pub lm_head: StableCell,
}

impl LlamaCached {
  /*pub fn make_deploy_input(&self) -> LanguageModelDeployIn {
    let ubat_sz = self.cfg.ubat_sz;
    let seq_cap = self.cfg.seq_cap;
    let in_tok = StableCell::array([seq_cap, ubat_sz], u16::dtype());
    LanguageModelDeployIn{in_tok}
  }*/

  pub fn init(&self) {
    // TODO
  }

  pub fn apply<X: Borrow<CellPtr>>(&self, in_tok: X) -> LanguageModelDeployOut {
    // TODO
    unimplemented!();
  }
}
