use crate::algo::fp::*;
use crate::cell::*;
use crate::librarium::lm::*;
use crate::op::*;
use crate::util::cell::*;
use crate::util::pickle::*;

use std::borrow::{Borrow};

#[derive(Clone, Copy, Debug)]
pub struct LlamaConfig {
  pub num_layer: i64,
  pub tok_dim: i64,
  pub head_dim: i64,
  pub num_head: i64,
  pub mlp_inner_dim: i64,
  pub seq_len: i64,
  pub ubat_sz: i64,
  pub rms_norm_eps: f32,
  pub dtype: Dtype,
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
  pub embed: StableCell,
  pub layers: Vec<LlamaLayer>,
  pub head_norm: StableCell,
  pub lm_head: StableCell,
}

impl From<LlamaConfig> for Llama {
  fn from(cfg: LlamaConfig) -> Llama {
    let ubat_sz = cfg.ubat_sz;
    let seq_len = cfg.seq_len;
    let num_head = cfg.num_head;
    let head_dim = cfg.head_dim;
    let inner_dim = num_head * head_dim;
    let mlp_inner_dim = cfg.mlp_inner_dim;
    let tok_dim = cfg.tok_dim;
    let num_layers = cfg.num_layer as usize;
    assert!(tok_dim <= u16::max_value() as i64 + 1);
    let embed = StableCell::array([tok_dim, inner_dim], f16::dtype());
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0 .. num_layers {
      let q = StableCell::array([inner_dim, inner_dim], f16::dtype());
      let k = StableCell::array([inner_dim, inner_dim], f16::dtype());
      let v = StableCell::array([inner_dim, inner_dim], f16::dtype());
      let o = StableCell::array([inner_dim, inner_dim], f16::dtype());
      let gate = StableCell::array([mlp_inner_dim, inner_dim], f16::dtype());
      let down = StableCell::array([inner_dim, mlp_inner_dim], f16::dtype());
      let up = StableCell::array([mlp_inner_dim, inner_dim], f16::dtype());
      let inv_freq = StableCell::array([head_dim / 2], f32::dtype());
      let pre_norm = StableCell::array([inner_dim], f16::dtype());
      let post_norm = StableCell::array([inner_dim], f16::dtype());
      layers.push(LlamaLayer{q, k, v, o, gate, down, up, inv_freq, pre_norm, post_norm});
    }
    let head_norm = StableCell::array([inner_dim], f16::dtype());
    let lm_head = StableCell::array([tok_dim, inner_dim], f16::dtype());
    Llama{cfg, embed, layers, head_norm, lm_head}
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
      matcher.insert((i, "mlp", "down"), self.layers[i].down.clone());
      matcher.insert((i, "mlp", "up"), self.layers[i].up.clone());
      matcher.insert((i, "inv_freq"), self.layers[i].inv_freq.clone());
      matcher.insert((i, "input_layernorm"), self.layers[i].pre_norm.clone());
      matcher.insert((i, "post_attention_layernorm"), self.layers[i].post_norm.clone());
    }
    matcher.insert("norm", self.head_norm.clone());
    matcher.insert("lm_head", self.lm_head.clone());
    let matches = matcher.match_(pickdir.clone_keys());
    matches.inv()
  }

  pub fn make_input(&self) -> LanguageModelIn {
    let ubat_sz = self.cfg.ubat_sz;
    let seq_len = self.cfg.seq_len;
    let in_tok = StableCell::array([ubat_sz, seq_len], u16::dtype());
    let in_lm_tok = StableCell::array([ubat_sz, seq_len], u16::dtype());
    LanguageModelIn{in_tok, in_lm_tok}
  }

  pub fn apply<X: Borrow<CellPtr>, Y: Borrow<CellPtr>>(&self, in_tok: X, in_lm_tok: Y) -> LanguageModelOut {
    fn block_causal_attention_mask(x: CellPtr) -> CellPtr {
      x.block_tri_elem_affine(1.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, -f32::inf())
    }
    fn rms_norm(x: &CellPtr, weight: &StableCell, eps: f32, dtype: Dtype) -> CellPtr {
      let m = x.cast(f32::dtype()).square().inner_mean();
      let x = x * (m + eps).rsqrt();
      (weight * x).cast(dtype)
    }
    let ubat_sz = self.cfg.ubat_sz;
    let seq_len = self.cfg.seq_len;
    let num_head = self.cfg.num_head;
    let head_dim = self.cfg.head_dim;
    let inner_dim = num_head * head_dim;
    let mlp_inner_dim = self.cfg.mlp_inner_dim;
    let tok_dim = self.cfg.tok_dim;
    let rms_norm_eps = self.cfg.rms_norm_eps;
    let mut stream = *in_tok.borrow();
    stream = stream.inner_one_hot(tok_dim, f16::dtype());
    stream = stream
            .new_shape([ubat_sz * seq_len, tok_dim])
            .block_mm([ubat_sz * seq_len, tok_dim], false, &self.embed, [tok_dim, inner_dim], false)
            .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    // TODO TODO
    for i in 0 .. 1 {
      // FIXME FIXME: layer norm.
      let pre_nrm = stream;
      //let pre_nrm = rms_norm(&stream, &self.layers[i].pre_norm, rms_norm_eps, f16::dtype());
      let q_proj = pre_nrm
                  .new_shape([ubat_sz * seq_len, num_head * head_dim])
                  .block_mm([ubat_sz * seq_len, inner_dim], false, &self.layers[i].q, [inner_dim, inner_dim], true)
                  .new_shape([ubat_sz, seq_len, num_head, head_dim]);
      let k_proj = pre_nrm
                  .new_shape([ubat_sz * seq_len, num_head * head_dim])
                  .block_mm([ubat_sz * seq_len, inner_dim], false, &self.layers[i].k, [inner_dim, inner_dim], true)
                  .new_shape([ubat_sz, seq_len, num_head, head_dim]);
      let v_proj = pre_nrm
                  .new_shape([ubat_sz * seq_len, num_head * head_dim])
                  .block_mm([ubat_sz * seq_len, inner_dim], false, &self.layers[i].v, [inner_dim, inner_dim], true)
                  .new_shape([ubat_sz, seq_len, num_head, head_dim]);
      // FIXME FIXME: rotary embedding.
      /*
      let (vcos, vsin) = rotational_embed(&v_proj, );
      let (q_proj, k_proj) = rotational_pos_embed((q_proj, k_proj, vcos, vsin, );
      */
      let q_proj = q_proj.new_shape([ubat_sz * seq_len, num_head * head_dim]);
      let k_proj = k_proj.new_shape([ubat_sz * seq_len, num_head * head_dim]);
      let attn = q_proj
                .block_mm_scale([seq_len, head_dim], false, k_proj, [seq_len, head_dim], true, 1.0 / (head_dim as f32).sqrt())
                .new_shape([ubat_sz, seq_len, num_head, seq_len])
                .cast(f32::dtype());
      let attn = block_causal_attention_mask(attn)
                .inner_softmax()
                .cast(f16::dtype())
                .new_shape([ubat_sz * seq_len, num_head * seq_len]);
      // FIXME FIXME: need pad for fp16 gemm.
      /*
      let v_proj = v_proj.new_shape([ubat_sz * seq_len, num_head * head_dim]);
      let v_attn = attn.block_mm([seq_len, seq_len], false, v_proj, [seq_len, head_dim], false);
      */
      let pad_head_dim = ((head_dim + 8 - 1) / 8) * 8;
      let v_proj = v_proj
                  .block_pad([seq_len, pad_head_dim], f16::zero())
                  .new_shape([ubat_sz * seq_len, num_head * pad_head_dim]);
      let v_attn = attn.block_mm([seq_len, seq_len], false, v_proj, [seq_len, pad_head_dim], false)
                  .new_shape([ubat_sz, seq_len, num_head, pad_head_dim])
                  .block_unpad([seq_len, head_dim])
                  .new_shape([ubat_sz * seq_len, num_head * head_dim]);
      let o_proj = v_attn.block_mm([ubat_sz * seq_len, inner_dim], false, &self.layers[0].o, [inner_dim, inner_dim], true)
                         .new_shape([ubat_sz, seq_len, num_head, head_dim]);
      stream = stream + o_proj;
      // FIXME FIXME: post layer norm, mlp.
      let post_nrm = stream;
      //let post_nrm = post_layer_norm(stream);
      //let post_nrm = rms_norm(&stream, &self.layers[i].post_norm, rms_norm_eps, f16::dtype());
      let up_proj = post_nrm
                   .new_shape([ubat_sz * seq_len, num_head * head_dim])
                   .block_mm([ubat_sz * seq_len, inner_dim], false, &self.layers[0].up, [mlp_inner_dim, inner_dim], true);
      let gate_proj = post_nrm
                     .new_shape([ubat_sz * seq_len, num_head * head_dim])
                     .block_mm([ubat_sz * seq_len, inner_dim], false, &self.layers[0].gate, [mlp_inner_dim, inner_dim], true);
      // FIXME: intermediate activation dtype.
      //let gate_proj = gate_proj.cast(f32::dtype()).standard_silu().cast(f16::dtype());
      let gate_proj = gate_proj.standard_silu();
      let gate_up = gate_proj * up_proj;
      let down_proj = gate_up
                     .block_mm([ubat_sz * seq_len, mlp_inner_dim], false, &self.layers[0].down, [inner_dim, mlp_inner_dim], true)
                     .new_shape([ubat_sz, seq_len, num_head, head_dim]);
      stream = stream + down_proj;
    }
    // TODO TODO
    //let stream = rms_norm(&stream, &self.head_norm, rms_norm_eps, f16::dtype());
    let out_lm_logit = stream
                      .new_shape([ubat_sz * seq_len, num_head * head_dim])
                      .block_mm([ubat_sz * seq_len, inner_dim], false, &self.lm_head, [tok_dim, inner_dim], true)
                      .new_shape([ubat_sz, seq_len, tok_dim])
                      .keep();
    let logit32 = out_lm_logit.cast(f32::dtype());
    let out_lm_prob = logit32.inner_softmax().keep();
    //let out_lm_loss = (-out_lm_prob.inner_select(in_lm_tok).ln()).keep();
    let out_lm_loss = logit32.inner_softmax_categorical_nll(in_lm_tok).keep();
    LanguageModelOut{out_lm_logit, out_lm_prob, out_lm_loss}
  }
}
