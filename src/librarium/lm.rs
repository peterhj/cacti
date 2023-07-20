use crate::prelude::*;

pub type LanguageModelIn = LanguageModelInput;

#[derive(Clone)]
pub struct LanguageModelInput {
  pub in_tok: StableCell,
  pub in_lm_tok: StableCell,
}

impl LanguageModelIn {
  pub fn deploy(&self) -> LanguageModelDeployIn {
    LanguageModelDeployIn{
      in_tok: self.in_tok.clone(),
    }
  }
}

pub type LanguageModelOut = LanguageModelOutput;

#[derive(Clone)]
pub struct LanguageModelOutput {
  pub out_lm_logit: StableCell,
  pub out_lm_prob: StableCell,
  pub out_lm_loss: StableCell,
}

impl LanguageModelOut {
  pub fn deploy(&self) -> LanguageModelDeployOut {
    LanguageModelDeployOut{
      out_lm_logit: self.out_lm_logit.clone(),
      out_lm_prob: self.out_lm_prob.clone(),
    }
  }
}

pub type LanguageModelDeployIn = LanguageModelDeployInput;

#[derive(Clone)]
pub struct LanguageModelDeployInput {
  pub in_tok: StableCell,
}

pub type LanguageModelDeployOut = LanguageModelDeployOutput;

#[derive(Clone)]
pub struct LanguageModelDeployOutput {
  pub out_lm_logit: StableCell,
  pub out_lm_prob: StableCell,
}
