extern crate cacti;

use cacti::prelude::*;
use cacti::algo::str::{safe_ascii};
use cacti::librarium::adamw::*;
use cacti::librarium::llama::*;
use cacti::librarium::sentencepiece::*;
use cacti::util::pickle::*;

use std::borrow::{Borrow};

fn main() {
  let data_dir = "data/openlm/open_llama_3b";
  let mut cfg  = LlamaConfig::open_llama_3b();
  //let data_dir = "data/openlm/open_llama_7b";
  //let mut cfg  = LlamaConfig::open_llama_7b();
  //let data_dir = "data/openlm/open_llama_13b";
  //let mut cfg  = LlamaConfig::open_llama_13b();

  cfg.ubat_sz = 1;
  cfg.seq_cap = 256;
  //cfg.seq_cap = 512;
  //cfg.seq_cap = 1024;
  println!("deploy: llama config: {:?}", cfg);

  let pickdir = PickleDir::open(data_dir).unwrap();
  println!("train:  loaded pickle dir");

  let tokenizer = SentencePieceTokenizer::from_dir(data_dir).unwrap();
  println!("train:  loaded sentencepiece tokenizer");
  println!("train:  tokenizer: n={:?}", tokenizer.num_pieces());
  println!("train:  tokenizer: unk={:?}", tokenizer.unk_id());
  println!("train:  tokenizer: bos={:?}", tokenizer.bos_id());
  println!("train:  tokenizer: eos={:?}", tokenizer.eos_id());
  println!("train:  tokenizer: pad={:?}", tokenizer.pad_id());

  let grad_scale = 1024.0_f32;
  let adamw = AdamW32{
    grad_unscale: 1.0 / grad_scale,
    //lr: 1.0e-5,
    lr: 2.0e-5,
    wd: 0.1,
    a1: 0.1,
    a2: 0.05,
    eps: 1.0e-5,
  };

  //let text_str = "Thucydides, an Athenian, wrote the history of";
  //let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its";
  let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its grounds. The preparations of both the combatants were in every department in the last state of perfection; and he could see the rest of the Hellenic race taking sides in the quarrel; those who delayed doing so at once having it in contemplation. Indeed this was the greatest movement yet known in history, not only of the Hellenes, but of a large part of the barbarian world-- I had almost said of mankind. For though the events of remote antiquity, and even those that more immediately preceded the war, could not from lapse of time be clearly ascertained, yet the evidences which an inquiry carried as far back as was practicable leads me to trust, all point to the conclusion that there was nothing on a great scale, either in war or in other matters.";
  let text_tok = tokenizer.encode16(text_str).unwrap();
  println!("train:  tokenizer: text str=\"{}\"", text_str);
  println!("train:  tokenizer: text str.len={}", text_str.len());
  println!("train:  tokenizer: text tok={:?}", text_tok.as_ref());
  println!("train:  tokenizer: text tok.len={}", text_tok.len());

  let mut model = Llama::from(cfg);
  let inv_matches = model.match_pickle_dir(&pickdir);
  let in_ = model.fresh_input();
  let in_tok = in_.in_tok;
  //println!("train:  matches: {:?}", &matches.mat);
  for &(ref cel, ref key) in inv_matches.mat.iter() {
    println!("train:  matches: key={:?} cel={:?}", key, cel);
  }
  let param = model.clone_param();
  let mut grad = Vec::with_capacity(param.len());
  for p in param.iter().rev() {
    //let g = StableCell::new();
    let g = StableCell::from(p.type_());
    //let g = StableCell::from(p.type_().cast(Dtype::Fp32));
    grad.push(g);
  }
  grad.reverse();
  /*for (p, g) in param.iter().zip(grad.iter()) {
    println!("train:  param={:?} grad={:?} ty={:?}", p, g, p.type_());
  }*/
  let mut master = Vec::with_capacity(param.len());
  let mut grad_avg = Vec::with_capacity(param.len());
  let mut grad2_avg = Vec::with_capacity(param.len());
  for p in param.iter() {
    let p_ = StableCell::from(p.type_().cast(f32::dtype_()));
    let g_ = StableCell::from(p.type_().cast(f32::dtype_()));
    let g2 = StableCell::from(p.type_().cast(f32::dtype_()));
    println!("train:  master={:?} ty={:?} grad avg={:?} ty={:?} grad2 avg={:?} ty={:?}",
        p_, p_.type_(), g_, g_.type_(), g2, g2.type_());
    master.push(p_);
    grad_avg.push(g_);
    grad2_avg.push(g2);
  }
  let mut param_log2_hist = Vec::with_capacity(param.len());
  let mut param_nan_count = Vec::with_capacity(param.len());
  let mut grad_log2_hist = Vec::with_capacity(param.len());
  let mut grad_nan_count = Vec::with_capacity(param.len());
  println!("train:  adamw: {:?}", adamw);
  println!("train:  grad scale: {:?}", grad_scale);
  //let loss_scale = (1.0_f32 / 256.0_f32) * 32.0_f32;
  //let loss_scale = (1.0_f32 / 256.0_f32) * grad_scale;
  let loss_scale = grad_scale / (text_tok.len() - 1) as f32;
  println!("train:  loss scale: {:?}", loss_scale);

  for cycle_nr in 0 .. 5 {
    println!("train:  start cycle={}", cycle_nr);

    reset();

    if cycle_nr == 0 {
      for (cel, _) in inv_matches.iter() {
        cel.mem_set_yield_();
      }
      smp_scope().with(|_| {
        for (p, p_) in param.iter().zip(master.iter()) {
          p_.set_cast(p);
        }
        for g_ in grad_avg.iter() {
          g_.set_zeros();
        }
        for g2 in grad2_avg.iter() {
          g2.set_zeros();
        }
      });
      model.init_constants();
    } else {
      smp_scope().with(|_| {
        for ((((g, g_), g2), p_), p) in
            grad.iter()
            .zip(grad_avg.iter())
            .zip(grad2_avg.iter())
            .zip(master.iter())
            .zip(param.iter())
            .rev()
        {
          // FIXME: zero out the zero-th row of the embed grad.
          if p == &model.embed {
            p.cache();
          } else {
            adamw.step(cycle_nr, &p_, &g_, &g2, &g);
            p.set_cast(p_.const_());
            //p.set_cast(p_);
          }
        }
      });
      model.cache_constants();
    }

    in_tok.mem_set_yield_();
    in_.in_lm_tok.mem_set_yield_();
    in_.in_lm_loss_scale.mem_set_yield_();

    let out = model.apply(&in_tok, &in_.in_lm_tok);

    println!("train:  in_lm_tok.shape={:?}", in_.in_lm_tok.shape());
    println!("train:  in_lm_tok.dtype={:?}", in_.in_lm_tok.dtype());
    println!("train:  in_lm_loss_scale.shape={:?}", in_.in_lm_loss_scale.shape());
    println!("train:  in_lm_loss_scale.dtype={:?}", in_.in_lm_loss_scale.dtype());
    println!("train:  out_lm_prob.shape={:?}", out.out_lm_prob.shape());
    println!("train:  out_lm_prob.dtype={:?}", out.out_lm_prob.dtype());
    println!("train:  out_lm_loss.shape={:?}", out.out_lm_loss.shape());
    println!("train:  out_lm_loss.dtype={:?}", out.out_lm_loss.dtype());

    let sink = CellMap::new();
    sink.add(&out.out_lm_loss, &in_.in_lm_loss_scale);
    let allsrc = CellMap::new();
    for (p, g) in param.iter().zip(grad.iter()) {
      allsrc.add(p, g);
    }
    allsrc.vadd_vjp(&sink);

    param_log2_hist.clear();
    param_nan_count.clear();
    grad_log2_hist.clear();
    grad_nan_count.clear();
    for (p, g) in param.iter().zip(grad.iter()) {
      if !(allsrc.get(p) == g) {
        println!("train:  WARNING: grad mismatch");
      }
      let p_log2_hist = p.abs_log2_hist8();
      param_log2_hist.push(p_log2_hist.keep());
      let p_nan_count = p.nan_count();
      param_nan_count.push(p_nan_count.keep());
      let g_log2_hist = g.abs_log2_hist8();
      grad_log2_hist.push(g_log2_hist.keep());
      let g_nan_count = g.nan_count();
      grad_nan_count.push(g_nan_count.keep());
    }

    compile();
    resume();

    let seq_cap = cfg.seq_cap;
    let tok_dim = cfg.tok_dim;
    if cycle_nr == 0 {
      for (cel, key) in inv_matches.iter() {
        let (pickty, pickfile) = pickdir.get(inv_matches.get(cel));
        resume_put(cel, &pickty, pickfile.mmap());
      }
    }
    resume_put_mem_with(&in_tok, |ty, mem| {
      println!("train:  set in_tok...");
      let mut tok_buf = Vec::with_capacity(seq_cap as _);
      tok_buf.push(1_u16);
      tok_buf.extend_from_slice(text_tok.as_ref());
      // FIXME: put end-of-sentence token.
      tok_buf.resize(seq_cap as _, 0_u16);
      let mem = ty.prepare_bytes_mut::<u16>(mem).unwrap();
      mem.copy_from_slice(&tok_buf);
    });
    resume_put_mem_with(&in_.in_lm_tok, |ty, mem| {
      println!("train:  set in_lm_tok...");
      let mut tok_buf = Vec::with_capacity(seq_cap as _);
      tok_buf.extend_from_slice(text_tok.as_ref());
      // FIXME: put end-of-sentence token.
      tok_buf.resize(seq_cap as _, 0_u16);
      let mem = ty.prepare_bytes_mut::<u16>(mem).unwrap();
      mem.copy_from_slice(&tok_buf);
    });
    resume_put_mem_with(&in_.in_lm_loss_scale, |ty, mem| {
      println!("train:  set in_lm_loss_scale...");
      let mut scale_buf = Vec::with_capacity(seq_cap as _);
      scale_buf.push(0.0_f32);
      scale_buf.resize(text_tok.len(), loss_scale);
      scale_buf.resize(seq_cap as _, 0.0_f32);
      let mem = ty.prepare_bytes_mut::<f32>(mem).unwrap();
      mem.copy_from_slice(&scale_buf);
    });
    for (idx, ((((g, g_), g2), p_), p)) in
        grad.iter()
        .zip(grad_avg.iter())
        .zip(grad2_avg.iter())
        .zip(master.iter())
        .zip(param.iter())
        .rev().enumerate()
    {
      if idx == 18 {
        println!("train:  dump grad avg: label={:?} post", inv_matches.get(p));
        let mem = g_._get_unsafe_mem();
        mem._debug_dump_f32();
      }
      if idx == 18 {
        println!("train:  dump master: label={:?} post", inv_matches.get(p));
        let mem = p_._get_unsafe_mem();
        mem._debug_dump_f32();
      }
      if idx == 18 {
        println!("train:  dump param: label={:?} post", inv_matches.get(p));
        let mem = p._get_unsafe_mem();
        mem._debug_dump_f16();
      }
    }
    let out_lm_prob_mem = out.out_lm_prob._get_unsafe_mem();
    let out_lm_loss_mem = out.out_lm_loss._get_unsafe_mem();
    println!("train:  out lm prob type   ={:?}", out.out_lm_prob.type_());
    println!("train:  out lm prob version={:?}", out.out_lm_prob.version());
    println!("train:  out lm prob mem ptr=0x{:016x}", out_lm_prob_mem.ptr as usize);
    println!("train:  out lm prob mem sz ={}", out_lm_prob_mem.sz);
    println!("train:  text str=\"{}\"", text_str);
    println!("train:  text str.len={}", text_str.len());
    println!("train:  text tok={:?}", text_tok.as_ref());
    println!("train:  text tok.len={}", text_tok.len());
    /*for pos in 0 .. text_tok.len() {
      let tok_id = text_tok.as_ref()[pos];
      println!("train:  text pos={} tok={} str={:?}",
          pos, tok_id, tokenizer.id_to_piece(tok_id as _).map(|s| safe_ascii(s.as_bytes())));
    }*/
    let out_lm_prob_f32 = out_lm_prob_mem._as_slice_f32();
    let out_lm_loss_f32 = out_lm_loss_mem._as_slice_f32();
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
    let mut loss_sum = 0.0;
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
      println!("train:  pos={} prev tok={} next tok={} max p={:.06} act p={:.06} loss={:.06} prev str={:?} next str={:?}",
          pos, prev_tok, arg_max, max_prob,
          (if (pos * ntok + act_next_tok as usize) < ((pos + 1) * ntok) {
            out_lm_prob_f32[pos * ntok + act_next_tok as usize]
          } else {
            -f32::inf()
          }),
          out_lm_loss_f32[pos as usize],
          tokenizer.id_to_piece(prev_tok as _).map(|s| safe_ascii(s.as_bytes())),
          tokenizer.id_to_piece(arg_max as _).map(|s| safe_ascii(s.as_bytes())),
      );
      if pos < text_tok.len() {
        loss_sum += out_lm_loss_f32[pos as usize];
      }
    }
    println!("train:  iter={} loss sum={:.06}", cycle_nr, loss_sum);
    // TODO
    //println!("train:  inspect param");
    for ((param, p_log2_hist), p_nan_count) in param.iter().zip(param_log2_hist.iter()).zip(param_nan_count.iter()) {
      let p_log2_hist_mem = p_log2_hist._get_unsafe_mem();
      if !(p_log2_hist_mem._as_slice_i64().len() == 0x100) {
        println!("train:  WARNING: param log2 hist: unexpected len: {}", p_log2_hist_mem._as_slice_i64().len());
        continue;
      }
      let p_nan_count_mem = p_nan_count._get_unsafe_mem();
      if !(p_nan_count_mem._as_slice_i64().len() == 1) {
        println!("train:  WARNING: param log2 hist: unexpected len: {}", p_nan_count_mem._as_slice_i64().len());
        continue;
      }
      let h = p_log2_hist_mem._as_slice_i64();
      let nan = p_nan_count_mem._as_slice_i64()[0];
      let mut total = 0;
      total += h[0];
      for &x in (&h[(0x7f_u8 - 24) as usize ..= (0x7f_u8 - 15) as usize]).iter() {
        total += x;
      }
      for &x in (&h[(0x7f_u8 - 14) as usize ..= (0x7f_u8 -  0) as usize]).iter() {
        total += x;
      }
      for &x in (&h[(0x7f_u8 +  1) as usize ..= (0x7f_u8 + 15) as usize]).iter() {
        total += x;
      }
      total += h[0xff];
      if h[0xff] != 0 {
      println!("train:  param log2 hist: zero={:?} sub={:?} -norm={:?} +norm={:?} unfin={:?} nan={:?} total={:?} label={:?}",
          h[0],
          &h[(0x7f_u8 - 24) as usize ..= (0x7f_u8 - 15) as usize],
          &h[(0x7f_u8 - 14) as usize ..= (0x7f_u8 -  0) as usize],
          &h[(0x7f_u8 +  1) as usize ..= (0x7f_u8 + 15) as usize],
          h[0xff],
          nan,
          total,
          inv_matches.get(param),
      );
        println!("train:  param log2 hist: WARNING: fp blowup: label={:?}",
            inv_matches.get(param),
        );
      }
    }
    //println!("train:  inspect gradient");
    for (((p, g), g_log2_hist), g_nan_count) in param.iter().zip(grad.iter()).zip(grad_log2_hist.iter()).zip(grad_nan_count.iter()) {
      if !(allsrc.get(p) == g) {
        println!("train:  WARNING: grad mismatch");
        continue;
      }
      let g_log2_hist_mem = g_log2_hist._get_unsafe_mem();
      if !(g_log2_hist_mem._as_slice_i64().len() == 0x100) {
        println!("train:  WARNING: grad log2 hist: unexpected len: {}", g_log2_hist_mem._as_slice_i64().len());
        continue;
      }
      let g_nan_count_mem = g_nan_count._get_unsafe_mem();
      if !(g_nan_count_mem._as_slice_i64().len() == 1) {
        println!("train:  WARNING: grad log2 hist: unexpected len: {}", g_nan_count_mem._as_slice_i64().len());
        continue;
      }
      let h = g_log2_hist_mem._as_slice_i64();
      let nan = g_nan_count_mem._as_slice_i64()[0];
      let mut total = 0;
      total += h[0];
      for &x in (&h[(0x7f_u8 - 24) as usize ..= (0x7f_u8 - 15) as usize]).iter() {
        total += x;
      }
      for &x in (&h[(0x7f_u8 - 14) as usize ..= (0x7f_u8 -  0) as usize]).iter() {
        total += x;
      }
      for &x in (&h[(0x7f_u8 +  1) as usize ..= (0x7f_u8 + 15) as usize]).iter() {
        total += x;
      }
      total += h[0xff];
      if h[0xff] != 0 {
      println!("train:  grad log2 hist: zero={:?} sub={:?} -norm={:?} +norm={:?} unfin={:?} nan={:?} total={:?} label={:?}",
          h[0],
          &h[(0x7f_u8 - 24) as usize ..= (0x7f_u8 - 15) as usize],
          &h[(0x7f_u8 - 14) as usize ..= (0x7f_u8 -  0) as usize],
          &h[(0x7f_u8 +  1) as usize ..= (0x7f_u8 + 15) as usize],
          h[0xff],
          nan,
          total,
          inv_matches.get(p),
      );
        println!("train:  param log2 hist: WARNING: fp blowup: label={:?}",
            inv_matches.get(p),
        );
      }
    }
    // TODO
    println!("train:  end cycle={}", cycle_nr);
    //break;
  }
}
