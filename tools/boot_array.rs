extern crate cacti;

use cacti::*;
use cacti::util::cell::*;
use cacti::util::pickle::*;

fn main() {
  let ubat_sz = 1;
  let seq_len = 512;
  //let seq_len = 2048;
  let tok_dim = 32000;
  let head_dim = 64;
  let num_head = 50;
  let inner_dim = head_dim * num_head;
  let mlp_inner_dim = 8640;
  let num_layer = 26;
  let mut pickdir = PickleDir::from("data/openlm/open_llama_3b");
  //let mut pickdir = PickleDir::from("data/openlm/open_llama_7b");
  //let mut pickdir = PickleDir::from("data/openlm/open_llama_13b");
  assert!(pickdir._reload().is_ok());
  let mut matcher = CellMatcher::new();
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
  let inv_matches = matches.inv();
  //println!("boot: matches: {:?}", &matches.mat);
  for &(ref cel, ref key) in inv_matches.mat.iter() {
    println!("boot: matches: key={:?} cel={:?}", key, cel);
  }
  for iter_nr in 0 .. 2 {
    reset();
    if iter_nr == 0 {
      //embed.mem_set_yield_();
      for (cel, _) in inv_matches.iter() {
        cel.mem_set_yield_();
      }
      //w.mem_set_yield_();
    } else {
      //embed.cache();
      for (cel, _) in inv_matches.iter() {
        cel.cache();
      }
      //w.cache();
    }
    in_tok.mem_set_yield_();
    let stream = in_tok.inner_one_hot(tok_dim, f16::dtype());
    let stream = stream.new_shape([ubat_sz * seq_len, tok_dim])
                       .block_mm([1, tok_dim], false, &embed, [tok_dim, inner_dim], false)
                       .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    // FIXME FIXME: layer norm.
    let prenrm = stream;
    //let prenrm = pre_layer_norm(&stream);
    let q_proj = prenrm.new_shape([ubat_sz * seq_len, num_head * head_dim])
                       .block_mm([1, inner_dim], false, &layers[0].q, [inner_dim, inner_dim], true)
                       .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    let k_proj = prenrm.new_shape([ubat_sz * seq_len, num_head * head_dim])
                       .block_mm([1, inner_dim], false, &layers[0].k, [inner_dim, inner_dim], true)
                       .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    let v_proj = prenrm.new_shape([ubat_sz * seq_len, num_head * head_dim])
                       .block_mm([1, inner_dim], false, &layers[0].v, [inner_dim, inner_dim], true)
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
    let o_proj = v_attn.block_mm([1, inner_dim], false, &layers[0].o, [inner_dim, inner_dim], true);
    let stream = stream + o_proj;
    // FIXME FIXME: post layer norm, mlp.
    //let stream = post_layer_norm(stream);
    let up_proj = stream.new_shape([ubat_sz * seq_len, num_head * head_dim])
                        .block_mm([1, inner_dim], false, &layers[0].up, [mlp_inner_dim, inner_dim], true);
    let gate_proj = stream.new_shape([ubat_sz * seq_len, num_head * head_dim])
                          .block_mm([1, inner_dim], false, &layers[0].gate, [mlp_inner_dim, inner_dim], true);
    //let gate_proj = activation(gate_proj);
    let gate_up = gate_proj * up_proj;
    let down_proj = gate_up.block_mm([1, mlp_inner_dim], false, &layers[0].down, [inner_dim, mlp_inner_dim], true)
                           .new_shape([ubat_sz, seq_len, num_head, head_dim]);
    let stream = stream + down_proj;
    /*
    // TODO
    // ...
    */
    let out_logit = stream.new_shape([ubat_sz * seq_len, num_head * head_dim])
                          .block_mm([1, inner_dim], false, &lm_head, [tok_dim, inner_dim], true)
                          .new_shape([ubat_sz, seq_len, tok_dim]);
    let out_prob = out_logit.inner_softmax();
    compile();
    resume();
    if iter_nr == 0 {
      //resume_put_mem_val(&embed, &0.0_f32);
      //resume_put_mem_val(&w, &0.0_f32);
      //resume_put_mem_fun(&embed, |_, mem| mem.copy_from_slice(&[0.0_f32]));
      /*resume_put_mem_fun(&embed, |ty, mem| {
        let (pickty, pickfile) = pickdir.open(inv_matches.get(&embed));
        assert_eq!(ty, pickty);
        mem.copy_from_reader(pickfile);
        mem._debug_dump_f16();
      });*/
      for (cel, key) in inv_matches.iter() {
        resume_put_mem_fun(cel, |ty, mem| {
          let (pickty, pickfile) = pickdir.open(inv_matches.get(cel));
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
      let mut tok_buf = Vec::with_capacity(seq_len as _);
      for _ in 0 .. seq_len {
        tok_buf.push(0_u16);
      }
      mem.copy_from_slice(&tok_buf);
    });
    // TODO
    break;
  }
}
