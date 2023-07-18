extern crate cacti;

use cacti::*;
use cacti::algo::str::*;
use cacti::librarium::adamw::*;
use cacti::librarium::llama::*;
use cacti::librarium::sentencepiece::*;
use cacti::util::cell::*;
use cacti::util::pickle::*;

fn main() {
  let mut cfg = LlamaConfig::open_llama_3b();
  //let mut cfg = LlamaConfig::open_llama_7b();
  //let mut cfg = LlamaConfig::open_llama_13b();
  cfg.seq_cap = 256;
  //cfg.seq_cap = 512;
  //cfg.seq_cap = 1024;
  let data_dir = "data/openlm/open_llama_3b";
  //let data_dir = "data/openlm/open_llama_7b";
  //let data_dir = "data/openlm/open_llama_13b";
  let pickdir = PickleDir::from(data_dir).unwrap();
  println!("boot: loaded pickle dir");
  let tokenizer = SentencePieceTokenizer::from_dir(data_dir).unwrap();
  println!("boot: loaded sentencepiece tokenizer");
  println!("boot: tokenizer: n={:?}", tokenizer.num_pieces());
  println!("boot: tokenizer: unk={:?}", tokenizer.unk_id());
  println!("boot: tokenizer: bos={:?}", tokenizer.bos_id());
  println!("boot: tokenizer: eos={:?}", tokenizer.eos_id());
  println!("boot: tokenizer: pad={:?}", tokenizer.pad_id());
  let adamw_cfg = AdamWConfig{
    lr: 1.0e-5,
    beta: 0.1,
    gamma: 0.001,
    eps: 1.0e-12,
  };
  //let text_str = "Thucydides, an Athenian, wrote the history of";
  //let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its";
  let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its grounds. The preparations of both the combatants were in every department in the last state of perfection; and he could see the rest of the Hellenic race taking sides in the quarrel; those who delayed doing so at once having it in contemplation. Indeed this was the greatest movement yet known in history, not only of the Hellenes, but of a large part of the barbarian world-- I had almost said of mankind. For though the events of remote antiquity, and even those that more immediately preceded the war, could not from lapse of time be clearly ascertained, yet the evidences which an inquiry carried as far back as was practicable leads me to trust, all point to the conclusion that there was nothing on a great scale, either in war or in other matters.";
  let text_tok = tokenizer.encode16(text_str).unwrap();
  println!("boot: tokenizer: text str=\"{}\"", text_str);
  println!("boot: tokenizer: text str.len={}", text_str.len());
  println!("boot: tokenizer: text tok={:?}", text_tok.as_ref());
  println!("boot: tokenizer: text tok.len={}", text_tok.len());
  let mut model = Llama::from(cfg);
  let inv_matches = model.match_pickle_dir(&pickdir);
  let input = model.make_input();
  let in_tok = input.in_tok;
  //println!("boot: matches: {:?}", &matches.mat);
  for &(ref cel, ref key) in inv_matches.mat.iter() {
    println!("boot: matches: key={:?} cel={:?}", key, cel);
  }
  let param = model.param();
  let mut grad = Vec::with_capacity(param.len());
  for p in param.iter().rev() {
    //let g = StableCell::new();
    let g = StableCell::from(p.type_());
    grad.push(g);
  }
  grad.reverse();
  for (p, g) in param.iter().zip(grad.iter()) {
    println!("boot: param={:?} grad={:?} ty={:?}", p, g, p.type_());
  }
  let mut adamw = AdamW::from(adamw_cfg);
  for iter_nr in 0 .. 1 {
    println!("boot: start iter...");
    reset();
    if iter_nr == 0 {
      for (cel, _) in inv_matches.iter() {
        cel.mem_set_yield_();
      }
      model.init();
    } else {
      for (cel, _) in inv_matches.iter() {
        cel.cache();
      }
      model.cache();
    }
    in_tok.mem_set_yield_();
    input.in_lm_tok.mem_set_yield_();
    let out = model.apply(&in_tok, &input.in_lm_tok);
    println!("boot: in_lm_tok.shape={:?}", input.in_lm_tok.shape());
    println!("boot: in_lm_tok.dtype={:?}", input.in_lm_tok.dtype());
    println!("boot: out_lm_prob.shape={:?}", out.out_lm_prob.shape());
    println!("boot: out_lm_prob.dtype={:?}", out.out_lm_prob.dtype());
    println!("boot: out_lm_loss.shape={:?}", out.out_lm_loss.shape());
    println!("boot: out_lm_loss.dtype={:?}", out.out_lm_loss.dtype());
    let sink = CellMap::new();
    sink.add(&out.out_lm_loss, out.out_lm_loss.type_().ones());
    // FIXME: stable grad.
    //let allsrc = sink.vjp();
    let allsrc = CellMap::new();
    allsrc.vadd(&param, &grad);
    vjp(&allsrc, &sink);
    /*
    // TODO TODO
    //let grad = allsrc.vget(&param);
    adamw.accumulate_grad(&param, &grad);
    if (iter_nr + 1) % minibatch_iter_ct == 0 {
      adamw.step(&param, &grad);
    }
    */
    compile();
    resume();
    let seq_cap = cfg.seq_cap;
    let tok_dim = cfg.tok_dim;
    if iter_nr == 0 {
      for (cel, key) in inv_matches.iter() {
        resume_put_mem_fun(cel, |ty, mem| {
          let (pickty, pickfile) = pickdir.get(inv_matches.get(cel));
          if ty != pickty {
            panic!("ERROR: type mismatch: cel={:?} key=\"{}\" ty={:?} pickty={:?}", cel, key, ty, pickty);
          }
          mem.copy_from_reader(pickfile);
          if ty.dtype == f16::dtype() {
            mem._debug_dump_f16();
          } else if ty.dtype == f32::dtype() {
            mem._debug_dump_f32();
          }
        });
      }
    }
    resume_put_mem_fun(&in_tok, |_, mem| {
      println!("boot: set in_tok...");
      let mut tok_buf = Vec::with_capacity(seq_cap as _);
      tok_buf.push(1_u16);
      tok_buf.extend_from_slice(text_tok.as_ref());
      tok_buf.resize(seq_cap as _, 0_u16);
      mem.copy_from_slice(&tok_buf);
    });
    resume_put_mem_fun(&input.in_lm_tok, |_, mem| {
      println!("boot: set in_lm_tok...");
      let mut tok_buf = Vec::with_capacity(seq_cap as _);
      for _ in 0 .. seq_cap {
        tok_buf.push(0_u16);
      }
      mem.copy_from_slice(&tok_buf);
    });
    let out_lm_prob_mem = out.out_lm_prob._get_mem();
    println!("boot: out lm prob type   ={:?}", out.out_lm_prob.type_());
    println!("boot: out lm prob version={:?}", out.out_lm_prob.version());
    println!("boot: out lm prob mem ptr=0x{:016x}", out_lm_prob_mem.ptr as usize);
    println!("boot: out lm prob mem sz ={}", out_lm_prob_mem.sz);
    println!("boot: text str=\"{}\"", text_str);
    println!("boot: text str.len={}", text_str.len());
    println!("boot: text tok={:?}", text_tok.as_ref());
    println!("boot: text tok.len={}", text_tok.len());
    /*for pos in 0 .. text_tok.len() {
      let tok_id = text_tok.as_ref()[pos];
      println!("boot: text pos={} tok={} str={:?}",
          pos, tok_id, tokenizer.id_to_piece(tok_id as _).map(|s| sane_ascii(s.as_bytes())));
    }*/
    let out_lm_prob_f32 = out_lm_prob_mem._as_slice_f32();
    fn argmax(xs: &[f32]) -> Option<(usize, f32)> {
      let mut kv = None;
      for (k, &v) in xs.iter().enumerate() {
        match kv {
          None => {
            kv = Some((k, v));
          }
          Some((_, ov)) => {
            if ov < v {
              kv = Some((k, v));
            }
          }
        }
      }
      kv
    }
    let ntok = tok_dim as usize;
    for pos in 1 ..= text_tok.len() {
      let prev_pos = pos - 1;
      let prev_tok = text_tok.as_ref()[prev_pos];
      let act_next_tok = if pos < text_tok.len() {
        text_tok.as_ref()[pos]
      } else {
        0
      };
      let (arg_max, max_prob) = argmax(&out_lm_prob_f32[pos * ntok .. (pos + 1) * ntok]).unwrap();
      println!("boot: pos={} prev tok={} next tok={} max p={:.06} act p={:.06} prev str={:?} next str={:?}",
          pos, prev_tok, arg_max, max_prob,
          (if (pos * ntok + act_next_tok as usize) < ((pos + 1) * ntok) {
            out_lm_prob_f32[pos * ntok + act_next_tok as usize]
          } else {
            -f32::inf()
          }),
          tokenizer.id_to_piece(prev_tok as _).map(|s| sane_ascii(s.as_bytes())),
          tokenizer.id_to_piece(arg_max as _).map(|s| sane_ascii(s.as_bytes())),
      );
    }
    // TODO
    println!("boot: end iter");
    //break;
  }
}
