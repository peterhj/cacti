extern crate cacti;
extern crate dessert;

use cacti::*;
use dessert::func::*;
use dessert::model::{Llama};
use dessert::opt::{AdamWConfig, AdamW};
use dessert::str_codec::{SentencePieceStrCodec};

fn main() {
  // Micro-batch size.
  let ubat_sz = 1;
  // Sequence capacity (i.e. maximum length).
  let seq_cap = 2048;

  let codec = SentencePieceStrCodec::open("tokenizer.model").unwrap();

  let llm = Llama::new_3b();
  //let llm = Llama::new_7b();
  //... etc.
  llm.load_torch_param("pytorch-model*.bin").unwrap();

  let mut opt = AdamW::new(Llama::default_adamw());
  opt.push_param(llm.param());

  //let x = StableCell::array([ubat_sz, seq_cap], Dtype::UInt16);
  let x = StableCell::array([ubat_sz, seq_cap], u16::dtype());
  let y = StableCell::array([ubat_sz, seq_cap], "u16");

  // TODO

  {
    reset();
    // FIXME: init.
    compile();
    resume();
  }

  loop {
    reset();
    for w in llm.param().iter() {
      // FIXME FIXME: think about this (w.cache()? w.cache_init()? ...).
      w.cache();
    }
    //x.set_mem(_);
    //y.set_mem(_);
    let logit = llm.apply(&x);
    let loss = batch_neg_likelihood(logit, &y);
    /*for w in llm.param().iter().rev() {
      w.gradl(loss).eval();
    }*/
    opt.apply_update(loss);
    compile();
    resume();
  }
}
