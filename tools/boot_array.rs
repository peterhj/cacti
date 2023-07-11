extern crate cacti;

use cacti::*;
use cacti::algo::str::*;
use cacti::librarium::llama::*;
use cacti::librarium::sentencepiece::*;
use cacti::util::cell::*;
use cacti::util::pickle::*;

fn main() {
  /*let ubat_sz = 1;
  let seq_cap = 512;
  //let seq_cap = 2048;
  let tok_dim = 32000;
  let head_dim = 100;
  let num_head = 32;
  let inner_dim = head_dim * num_head;
  let mlp_inner_dim = 8640;
  let num_layer = 26;
  let rms_norm_eps = 1.0e-6_f32;
  let dtype = f16::dtype();*/
  let mut cfg = LlamaConfig::open_llama_3b();
  //let cfg = LlamaConfig::open_llama_7b();
  //let cfg = LlamaConfig::open_llama_13b();
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
  //let text_str = "Thucydides, an Athenian, wrote the history of";
  //let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its";
  let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its grounds. The preparations of both the combatants were in every department in the last state of perfection; and he could see the rest of the Hellenic race taking sides in the quarrel; those who delayed doing so at once having it in contemplation. Indeed this was the greatest movement yet known in history, not only of the Hellenes, but of a large part of the barbarian world-- I had almost said of mankind. For though the events of remote antiquity, and even those that more immediately preceded the war, could not from lapse of time be clearly ascertained, yet the evidences which an inquiry carried as far back as was practicable leads me to trust, all point to the conclusion that there was nothing on a great scale, either in war or in other matters.";
  let text_tok = tokenizer.encode16(text_str).unwrap();
  println!("boot: tokenizer: text str=\"{}\"", text_str);
  println!("boot: tokenizer: text str.len={}", text_str.len());
  println!("boot: tokenizer: text tok={:?}", text_tok.as_ref());
  println!("boot: tokenizer: text tok.len={}", text_tok.len());
  /*let mut matcher = CellMatcher::new();
  let in_tok = StableCell::array([ubat_sz, seq_len], u16::dtype());
  //let out_next_tok_logit = StableCell::array([ubat_sz, seq_len, tok_dim], f32::dtype());
  //let out_next_tok_prob = StableCell::array([ubat_sz, seq_len, tok_dim], f32::dtype());
  //let out_model_loss = StableCell::array([ubat_sz, seq_len], f32::dtype());
  let embed = StableCell::array([tok_dim, inner_dim], f16::dtype());
  matcher.insert("embed", embed.clone());
  //let w = StableCell::array([inner_dim, inner_dim], f32::dtype());
  //let w = StableCell::array([inner_dim, inner_dim], bf16::dtype());
  fn block_causal_attention_mask(x: CellPtr) -> CellPtr {
    x
    //unimplemented!();
  }
  struct LlamaLayer {
    in_norm: StableCell,
    inv_freq: StableCell,
    q: StableCell,
    k: StableCell,
    v: StableCell,
    o: StableCell,
    post_norm: StableCell,
    gate: StableCell,
    up: StableCell,
    down: StableCell,
  }
  let mut layers = Vec::new();
  for layer_idx in 0 .. num_layer {
    let q = StableCell::array([inner_dim, inner_dim], f16::dtype());
    let k = StableCell::array([inner_dim, inner_dim], f16::dtype());
    let v = StableCell::array([inner_dim, inner_dim], f16::dtype());
    let o = StableCell::array([inner_dim, inner_dim], f16::dtype());
    matcher.insert((layer_idx, "attn", "q_proj"), q.clone());
    matcher.insert((layer_idx, "attn", "k_proj"), k.clone());
    matcher.insert((layer_idx, "attn", "v_proj"), v.clone());
    matcher.insert((layer_idx, "attn", "o_proj"), o.clone());
    let inv_freq = StableCell::array([num_head], f32::dtype());
    matcher.insert((layer_idx, "inv_freq"), inv_freq.clone());
    let gate = StableCell::array([mlp_inner_dim, inner_dim], f16::dtype());
    let down = StableCell::array([inner_dim, mlp_inner_dim], f16::dtype());
    let up = StableCell::array([mlp_inner_dim, inner_dim], f16::dtype());
    matcher.insert((layer_idx, "mlp", "gate"), gate.clone());
    matcher.insert((layer_idx, "mlp", "down"), down.clone());
    matcher.insert((layer_idx, "mlp", "up"), up.clone());
    let in_norm = StableCell::array([inner_dim], f16::dtype());
    let post_norm = StableCell::array([inner_dim], f16::dtype());
    matcher.insert((layer_idx, "input_layernorm"), in_norm.clone());
    matcher.insert((layer_idx, "post_attention_layernorm"), post_norm.clone());
    layers.push(LlamaLayer{q, k, v, o, inv_freq, gate, down, up, in_norm, post_norm});
  }
  let head_norm = StableCell::array([inner_dim], f16::dtype());
  let lm_head = StableCell::array([tok_dim, inner_dim], f16::dtype());
  matcher.insert("norm", head_norm.clone());
  matcher.insert("lm_head", lm_head.clone());
  let matches = matcher.match_(pickdir.clone_keys());
  let inv_matches = matches.inv();*/
  /*let cfg = LlamaConfig{
    /*ubat_sz: 1,
    seq_len: 512,
    tok_dim: 32000,
    head_dim: 64,
    num_head: 50,
    mlp_inner_dim: 8640,
    num_layer: 26,*/
    ubat_sz,
    seq_cap,
    tok_dim,
    head_dim,
    num_head,
    mlp_inner_dim,
    num_layer,
    rms_norm_eps,
    dtype,
  };*/
  let mut model = Llama::from(cfg);
  let inv_matches = model.match_pickle_dir(&pickdir);
  let input = model.make_input();
  let in_tok = input.in_tok;
  //println!("boot: matches: {:?}", &matches.mat);
  for &(ref cel, ref key) in inv_matches.mat.iter() {
    println!("boot: matches: key={:?} cel={:?}", key, cel);
  }
  for iter_nr in 0 .. 1 {
    println!("boot: start iter...");
    reset();
    if iter_nr == 0 {
      //embed.mem_set_yield_();
      for (cel, _) in inv_matches.iter() {
        cel.mem_set_yield_();
      }
      //w.mem_set_yield_();
      model.init();
    } else {
      //embed.cache();
      for (cel, _) in inv_matches.iter() {
        cel.cache();
      }
      //w.cache();
      model.cache();
    }
    in_tok.mem_set_yield_();
    input.in_lm_tok.mem_set_yield_();
    /*let stream = in_tok.inner_one_hot(tok_dim, f16::dtype());
    let stream = stream.new_shape([ubat_sz * seq_len, tok_dim])
                       .block_mm([ubat_sz * seq_len, tok_dim], false, &model.embed, [tok_dim, inner_dim], false)
                       .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    // FIXME FIXME: layer norm.
    let prenrm = stream;
    //let prenrm = pre_layer_norm(&stream);
    let q_proj = prenrm.new_shape([ubat_sz * seq_len, num_head * head_dim])
                       .block_mm([ubat_sz * seq_len, inner_dim], false, &model.layers[0].q, [inner_dim, inner_dim], true)
                       .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    let k_proj = prenrm.new_shape([ubat_sz * seq_len, num_head * head_dim])
                       .block_mm([ubat_sz * seq_len, inner_dim], false, &model.layers[0].k, [inner_dim, inner_dim], true)
                       .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    let v_proj = prenrm.new_shape([ubat_sz * seq_len, num_head * head_dim])
                       .block_mm([ubat_sz * seq_len, inner_dim], false, &model.layers[0].v, [inner_dim, inner_dim], true)
                       .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    // FIXME FIXME: rotary embedding.
    /*
    let (vcos, vsin) = rotational_embed(&v_proj, );
    let (q_proj, k_proj) = rotational_pos_embed((q_proj, k_proj, vcos, vsin, );
    */
    let q_proj = q_proj.new_shape([ubat_sz * seq_len, num_head * head_dim]);
    let k_proj = k_proj.new_shape([ubat_sz * seq_len, num_head * head_dim]);
    let attn = q_proj.block_mm_scale([seq_len, head_dim], false, k_proj, [seq_len, head_dim], true, 1.0 / (head_dim as f32).sqrt())
              .new_shape([ubat_sz, seq_len, num_head, seq_len])
              .cast(f32::dtype());
    let attn = block_causal_attention_mask(attn)
              .inner_softmax()
              .cast(f16::dtype())
              .new_shape([ubat_sz * seq_len, num_head * seq_len]);
    let v_proj = v_proj.new_shape([ubat_sz * seq_len, num_head * head_dim]);
    let v_attn = attn.block_mm([seq_len, seq_len], false, v_proj, [seq_len, head_dim], false);
    let o_proj = v_attn.block_mm([ubat_sz * seq_len, inner_dim], false, &model.layers[0].o, [inner_dim, inner_dim], true)
                       .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    let stream = stream + o_proj;
    // FIXME FIXME: post layer norm, mlp.
    //let stream = post_layer_norm(stream);
    let up_proj = stream.new_shape([ubat_sz * seq_len, num_head * head_dim])
                        .block_mm([ubat_sz * seq_len, inner_dim], false, &model.layers[0].up, [mlp_inner_dim, inner_dim], true);
    let gate_proj = stream.new_shape([ubat_sz * seq_len, num_head * head_dim])
                          .block_mm([ubat_sz * seq_len, inner_dim], false, &model.layers[0].gate, [mlp_inner_dim, inner_dim], true);
    //let gate_proj = activation(gate_proj);
    let gate_up = gate_proj * up_proj;
    let down_proj = gate_up.block_mm([ubat_sz * seq_len, mlp_inner_dim], false, &model.layers[0].down, [inner_dim, mlp_inner_dim], true)
                           .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    let stream = stream + down_proj;
    /*
    // TODO
    // ...
    */
    let out_lm_logit = stream.new_shape([ubat_sz * seq_len, num_head * head_dim])
                             .block_mm([1, inner_dim], false, &model.lm_head, [tok_dim, inner_dim], true)
                             .new_shape([ubat_sz, seq_len, tok_dim]);
    let out_lm_prob = out_lm_logit.inner_softmax();*/
    let out = model.apply(&in_tok, &input.in_lm_tok);
    println!("boot: in_lm_tok.shape={:?}", input.in_lm_tok.shape());
    println!("boot: in_lm_tok.dtype={:?}", input.in_lm_tok.dtype());
    println!("boot: out_lm_prob.shape={:?}", out.out_lm_prob.shape());
    println!("boot: out_lm_prob.dtype={:?}", out.out_lm_prob.dtype());
    println!("boot: out_lm_loss.shape={:?}", out.out_lm_loss.shape());
    println!("boot: out_lm_loss.dtype={:?}", out.out_lm_loss.dtype());
    compile();
    resume();
    let seq_cap = cfg.seq_cap;
    let tok_dim = cfg.tok_dim;
    if iter_nr == 0 {
      //resume_put_mem_val(&embed, &0.0_f32);
      //resume_put_mem_val(&w, &0.0_f32);
      //resume_put_mem_fun(&embed, |_, mem| mem.copy_from_slice(&[0.0_f32]));
      /*resume_put_mem_fun(&embed, |ty, mem| {
        let (pickty, pickfile) = pickdir.get(inv_matches.get(&embed));
        assert_eq!(ty, pickty);
        mem.copy_from_reader(pickfile);
        mem._debug_dump_f16();
      });*/
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
      //resume_put_mem_fun(&w, |_, mem| mem.copy_from_slice(&[0.0_f32]));
    }
    resume_put_mem_fun(&in_tok, |_, mem| {
      println!("boot: set in_tok...");
      let mut tok_buf = Vec::with_capacity(seq_cap as _);
      /*for _ in 0 .. seq_cap {
        // FIXME: this should cause failure.
        //tok_buf.push(50000_u16);
        tok_buf.push(0_u16);
      }*/
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
    let out_lm_prob_mem = out.out_lm_prob.get_mem();
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
