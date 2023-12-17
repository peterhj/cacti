use crate::prelude::*;
use crate::librarium::llama::*;

pub type MistralConfig = LlamaConfig;

impl MistralConfig {
  pub fn mistral_7b() -> MistralConfig {
    MistralConfig{
      num_layer:  32,
      tok_dim:    32000,
      head_dim:   128,
      num_head:   32,
      q_group:    4,
      mlp_inner_dim:  14336,
      pos_len:    4096,
      rms_norm_eps:   1.0e-5,
      param_scale:    None,
      rope_lin_scale: None,
      ten_par:    None,
      seq_cap:    4096,
      ubat_sz:    1,
      dtype:      Dtype::Bf16,
    }
  }
}
