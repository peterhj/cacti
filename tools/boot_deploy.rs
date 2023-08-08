extern crate cacti;

use cacti::*;
use cacti::algo::str::*;
use cacti::librarium::adamw::*;
use cacti::librarium::llama::*;
use cacti::librarium::sentencepiece::*;
use cacti::util::cell::*;
use cacti::util::pickle::*;

use std::borrow::{Borrow};

fn main() {
  let data_dir = "data/openlm/open_llama_3b";
  //let data_dir = "data/openlm/open_llama_7b";
  //let data_dir = "data/openlm/open_llama_13b";
  let mut cfg = LlamaConfig::open_llama_3b();
  //let mut cfg = LlamaConfig::open_llama_7b();
  //let mut cfg = LlamaConfig::open_llama_13b();
  cfg.ubat_sz = 1;
  cfg.seq_cap = 256;
  //cfg.seq_cap = 512;
  //cfg.seq_cap = 1024;
  let pickdir = PickleDir::from(data_dir).unwrap();
  println!("boot: loaded pickle dir");
  let tokenizer = SentencePieceTokenizer::from_dir(data_dir).unwrap();
  println!("boot: loaded sentencepiece tokenizer");
  println!("boot: tokenizer: n={:?}", tokenizer.num_pieces());
  println!("boot: tokenizer: unk={:?}", tokenizer.unk_id());
  println!("boot: tokenizer: bos={:?}", tokenizer.bos_id());
  println!("boot: tokenizer: eos={:?}", tokenizer.eos_id());
  println!("boot: tokenizer: pad={:?}", tokenizer.pad_id());
  let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its grounds. The preparations of both the combatants were in every department in the last state of perfection; and he could see the rest of the Hellenic race taking sides in the quarrel; those who delayed doing so at once having it in contemplation. Indeed this was the greatest movement yet known in history, not only of the";
  //let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its grounds. The preparations of both the combatants were in every department in the last state of perfection; and he could see the rest of the Hellenic race taking sides in the quarrel; those who delayed doing so at once having it in contemplation. Indeed this was the greatest movement yet known in history, not only of the Hellenes, but of a large part of the barbarian world-- I had almost said of mankind. For though the events of remote antiquity, and even those that more immediately preceded the war, could not from lapse of time be clearly ascertained, yet the evidences which an inquiry carried as far back as was practicable leads me to trust, all point to the conclusion that there was nothing on a great scale, either in war or in other matters.";
  let text_tok = tokenizer.encode16(text_str).unwrap();
  println!("boot: tokenizer: text str=\"{}\"", text_str);
  println!("boot: tokenizer: text str.len={}", text_str.len());
  println!("boot: tokenizer: text tok={:?}", text_tok.as_ref());
  println!("boot: tokenizer: text tok.len={}", text_tok.len());
  let mut model = LlamaCached::from(cfg);
  let inv_matches = model.match_pickle_dir(&pickdir);
  let mut in_ = model.fresh_input();
  //println!("boot: matches: {:?}", &matches.mat);
  for &(ref cel, ref key) in inv_matches.mat.iter() {
    println!("boot: matches: key={:?} cel={:?}", key, cel);
  }
  let param = model.clone_param();
  for iter_nr in 0 .. 40 {
    println!("boot: start iter={}", iter_nr);
    reset();
    if iter_nr == 0 {
      for (cel, _) in inv_matches.iter() {
        cel.mem_set_yield_();
      }
      model.init_constants();
      model.init_state();
    } else {
      /*for p in param.iter() {
        p.cache();
      }*/
      model.cache_constants();
      model.cache_state();
      /*model.reset_state();*/
    }
    if iter_nr == 0 {
      in_[0].in_tok.mem_set_yield_();
    } else {
      in_[0].in_tok.cache();
    }
    let out = if iter_nr == 0 {
      model.apply(&mut in_, 0, 1 + text_tok.len() as i64)
    } else {
      model.apply(&mut in_, 1 + text_tok.len() as i64 + iter_nr - 1, 1 + text_tok.len() as i64 + iter_nr)
    };
    println!("boot: in_tok.shape={:?}", in_[0].in_tok.shape());
    println!("boot: in_tok.dtype={:?}", in_[0].in_tok.dtype());
    println!("boot: out_lm_logit.shape={:?}", out[0].out_lm_logit.shape());
    println!("boot: out_lm_logit.dtype={:?}", out[0].out_lm_logit.dtype());
    println!("boot: out_lm_prob.shape={:?}", out[0].out_lm_prob.shape());
    println!("boot: out_lm_prob.dtype={:?}", out[0].out_lm_prob.dtype());
    println!("boot: out_lm_tok.shape={:?}", out[0].out_lm_tok.shape());
    println!("boot: out_lm_tok.dtype={:?}", out[0].out_lm_tok.dtype());
    compile();
    resume();
    let seq_cap = cfg.seq_cap;
    let tok_dim = cfg.tok_dim;
    if iter_nr == 0 {
      for (cel, key) in inv_matches.iter() {
        resume_put_mem_with(cel, |ty, mem| {
          let (pickty, pickfile) = pickdir.get(inv_matches.get(cel));
          if ty.unbroadcast() != pickty.unbroadcast() {
            panic!("ERROR: type mismatch: cel={:?} key=\"{}\" ty={:?} pickty={:?}", cel, key, ty, pickty);
          }
          mem.copy_from_reader(pickfile);
        });
      }
    }
    if iter_nr == 0 {
      resume_put_mem_with(&in_[0].in_tok, |_, mem| {
        println!("boot: set in_tok...");
        let mut tok_buf = Vec::with_capacity(seq_cap as _);
        tok_buf.push(1_u16);
        tok_buf.extend_from_slice(text_tok.as_ref());
        // FIXME: put end-of-sentence token.
        tok_buf.resize(seq_cap as _, 0_u16);
        mem.copy_from_slice(&tok_buf);
      });
    }
    let in_tok_mem = in_[0].in_tok._get_mem();
    let out_tok_mem = out[0].out_lm_tok._get_mem();
    //let out_lm_prob_mem = out[0].out_lm_prob._get_mem();
    //println!("boot: in tok type   ={:?}", in_[0].in_tok.type_());
    //println!("boot: in tok version={:?}", in_[0].in_tok.version());
    println!("boot: in tok mem ptr=0x{:016x}", in_tok_mem.ptr as usize);
    println!("boot: in tok mem sz ={}", in_tok_mem.sz);
    println!("boot: out tok mem ptr=0x{:016x}", out_tok_mem.ptr as usize);
    println!("boot: out tok mem sz ={}", out_tok_mem.sz);
    /*println!("boot: out lm prob type   ={:?}", out[0].out_lm_prob.type_());
    println!("boot: out lm prob version={:?}", out[0].out_lm_prob.version());
    println!("boot: out lm prob mem ptr=0x{:016x}", out_lm_prob_mem.ptr as usize);
    println!("boot: out lm prob mem sz ={}", out_lm_prob_mem.sz);*/
    if iter_nr == 0 {
      println!("boot: text str=\"{}\"", text_str);
      println!("boot: text str.len={}", text_str.len());
      println!("boot: text tok={:?}", text_tok.as_ref());
      println!("boot: text tok.len={}", text_tok.len());
    }
    let in_tok_u16 = in_tok_mem._as_slice_u16();
    let out_tok_u16 = out_tok_mem._as_slice_u16();
    //let ntok = tok_dim as usize;
    let (start_pos, fin_pos) = if iter_nr == 0 {
      (1, 1 + text_tok.len())
    } else {
      (1 + text_tok.len() + (iter_nr) as usize, 1 + text_tok.len() + (iter_nr) as usize)
    };
    for pos in start_pos ..= fin_pos {
      let prev_pos = pos - 1;
      let act_prev_tok = if prev_pos >= 1 && prev_pos < text_tok.len() + 1 {
        text_tok.as_ref()[prev_pos - 1]
      } else {
        0
      };
      let prev_tok = in_tok_u16[prev_pos];
      /*let act_next_tok = if pos < text_tok.len() + 1 {
        text_tok.as_ref()[pos - 1]
      } else {
        0
      };*/
      let next_tok = out_tok_u16[pos - start_pos];
      let act_next_tok = in_tok_u16[pos];
      println!("boot: pos={} act prev={} prev={} next={} act next={} act prev={:?} prev={:?} next={:?} act next={:?}",
          pos, act_prev_tok, prev_tok, next_tok, act_next_tok,
          tokenizer.id_to_piece(act_prev_tok as _).map(|s| sane_ascii(s.as_bytes())),
          tokenizer.id_to_piece(prev_tok as _).map(|s| sane_ascii(s.as_bytes())),
          tokenizer.id_to_piece(next_tok as _).map(|s| sane_ascii(s.as_bytes())),
          tokenizer.id_to_piece(act_next_tok as _).map(|s| sane_ascii(s.as_bytes())),
      );
    }
    // TODO
    println!("boot: end iter={}", iter_nr);
  }
}
