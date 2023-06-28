use crate::cell::*;

pub type LanguageModelIn = LanguageModelInput;

#[derive(Clone)]
pub struct LanguageModelInput {
  pub in_tok: StableCell,
  pub in_lm_tok: StableCell,
}

pub type LanguageModelOut = LanguageModelOutput;

#[derive(Clone)]
pub struct LanguageModelOutput {
  pub out_lm_logit: StableCell,
  pub out_lm_prob: StableCell,
  pub out_lm_loss: StableCell,
}
