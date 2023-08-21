extern crate cacti;

use cacti::prelude::*;
use cacti::algo::str::{safe_ascii};
use cacti::librarium::adamw::*;
use cacti::librarium::llama::*;
use cacti::librarium::sentencepiece::*;
use cacti::util::pickle::*;

fn main() {
  // To run this fine-tuning example, you will need to have
  // a copy of OpenLLaMA 3B in a local directory.
  // The remote repo url is:
  //
  //    https://huggingface.co/openlm-research/open_llama_3b_v2
  //
  // Once you have cloned the repo and downloaded the parameters
  // via git LFS, please set `data_dir` below to the local path
  // at which you cloned the repo.
  //
  // This example was written with OpenLLaMA 3B in mind,
  // but you are encouraged to try some other models.
  let data_dir = "data/openlm/open_llama_3b_v2";
  let mut cfg  = LlamaConfig::open_llama_3b();

  // Set the micro-batch size to 1.
  cfg.ubat_sz = 1;

  // Also set the maximum sequence length to 256, which will
  // be all we need for the example below.
  //
  // You will need to increase this if your training examples
  // are sequences than 256.
  cfg.seq_cap = 256;

  println!("deploy: llama config: {:?}", cfg);

  // Set the AdamW minibatch size.
  //
  // (For this toy example, we are just fine-tuning on
  // a single sequence, so our minibatch size is just 1.)
  let minibatch_sz: i32 = 1;
  //let minibatch_sz: i32 = 2;

  // Check that the micro-batch size evenly divides the
  // minibatch size.
  assert_eq!(minibatch_sz % cfg.ubat_sz as i32, 0);

  // The total number of minibatches to train for.
  //
  // (For this toy example, 1 minibatch = 1 epoch.
  // When doing real training/fine-tuning, you will
  // likely want to limit the epochs to just 1 or 2.)
  let num_minibatch_iters: i32 = 4;

  // FP16 training can go off the rails if the gradient
  // scaling is insufficient.
  //
  // Here, we manually set a constant multiplicative
  // factor to scale the loss (and, by extension, the
  // gradients).
  //
  // You'll likely want to tune this for your own data,
  // and possibly also use auto-scaling (a topic beyond
  // this example).
  let grad_scale = 1024.0_f32;

  // Configure AdamW hyperparameters.
  //
  // Note that `grad_unscale` should be the reciprocal of
  // `grad_scale` (see above).
  let adamw = AdamW32{
    grad_unscale: 1.0 / grad_scale,
    //lr: 1.0e-5,
    lr: 2.0e-5,
    wd: 0.1,
    a1: 0.1,
    a2: 0.05,
    eps: 1.0e-5,
  };

  // Summarize the AdamW hyperparameters, including our
  // chosen gradient scaling.
  println!("train:  adamw: {:?}", adamw);
  println!("train:  grad scale: {:?}", grad_scale);

  // The directory we provided at the beginning of `main`
  // should contain "pytorch_model*.bin" pickle files.
  // Opening a `PickleDir` will safely parse the files'
  // metadata, without loading all the data just yet.
  let pickdir = PickleDir::open(data_dir).unwrap();
  println!("train:  loaded pickle dir");

  // Also load a SentencePiece tokenizer from the
  // "tokenizer.model" file in the above directory.
  let tokenizer = SentencePieceTokenizer::from_dir(data_dir).unwrap();
  println!("train:  loaded sentencepiece tokenizer");
  println!("train:  tokenizer: n={:?}", tokenizer.num_pieces());
  println!("train:  tokenizer: unk={:?}", tokenizer.unk_id());
  println!("train:  tokenizer: bos={:?}", tokenizer.bos_id());
  println!("train:  tokenizer: eos={:?}", tokenizer.eos_id());
  println!("train:  tokenizer: pad={:?}", tokenizer.pad_id());

  // And this will be our toy example for inference.
  // A real classic. (A 19th century translation by
  // Richard Crawley, via Project Gutenberg.)
  let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its grounds. The preparations of both the combatants were in every department in the last state of perfection; and he could see the rest of the Hellenic race taking sides in the quarrel; those who delayed doing so at once having it in contemplation. Indeed this was the greatest movement yet known in history, not only of the Hellenes, but of a large part of the barbarian world-- I had almost said of mankind. For though the events of remote antiquity, and even those that more immediately preceded the war, could not from lapse of time be clearly ascertained, yet the evidences which an inquiry carried as far back as was practicable leads me to trust, all point to the conclusion that there was nothing on a great scale, either in war or in other matters.";
  println!("train:  tokenizer: text str=\"{}\"", text_str);
  println!("train:  tokenizer: text str.len={}", text_str.len());

  // Now let's tokenize that string!
  // We should get roughly 220 16-bit tokens.
  let text_tok = tokenizer.encode16(text_str).unwrap();
  println!("train:  tokenizer: text tok={:?}", text_tok.as_ref());
  println!("train:  tokenizer: text tok.len={}", text_tok.len());

  // `Llama::from` will create a LLaMA model from the
  // provided config that is suitable for inference.
  //
  // `Llama` is implemented in (src/librarium/llama.rs),
  // which you may also want to read.
  let mut model = Llama::from(cfg);

  // So, we have an in-memory `model`, and we have a
  // `pickdir` that represents the contents on disk.
  // We will need to connect the two in order to figure out
  // which on-disk pickle tensors map to which in-memory
  // cells and vice versa. To do so, we use a convenience
  // function, `Llama::match_pickle_dir`, which itself
  // is implemented using parts from (src/util/cell.rs) and
  // (src/util/safepickle.rs).
  let inv_matches = model.match_pickle_dir(&pickdir);
  for (cel, key) in inv_matches.iter() {
    println!("train:  matches: key={:?} cel={:?}", key, cel);
  }

  // Create a fresh `Vec` of language model input cells,
  // where the length of the `Vec` is the micro-batch size.
  // The full type of `in_` is `Vec<LanguageModelDeployInput>`;
  // see (src/librarium/lm.rs).
  let in_ = model.fresh_input();

  // Clone the `Vec` of model parameters.
  // Note that only _references_ to the parameters are
  // cloned, so this is a cheap operation.
  let param = model.clone_param();

  // Here we will pre-define the dataflow cells that are the
  // gradients corresponding to the model parameters.
  // In particular, these gradient cells will be `StableCell`'s.
  //
  // A `StableCell` is a dataflow cell that persists across
  // `reset`'s. A `StableCell` is also (mostly) left alone by
  // run-time garbage collection; there will always be at least
  // one replica of a `StableCell` in GPU VRAM or host CPU RAM,
  // but not necessarily both at the same time.
  //
  // (It's not strictly necessary to reverse the order of the
  // gradient cells here; reversing the order is mostly useful
  // only for debug output.)
  let mut grad = Vec::with_capacity(param.len());
  for p in param.iter().rev() {
    let g = StableCell::from(p.type_());
    // In a future release of `cacti`, we would like to support
    // gradients in a higher precision than the parameters
    // (e.g. FP16 parameters with FP32 gradients).
    /*let g = StableCell::from(p.type_().cast(Dtype::Fp32));*/
    grad.push(g);
  }
  grad.reverse();
  /*for (p, g) in param.iter().zip(grad.iter()) {
    println!("train:  param={:?} grad={:?} ty={:?}", p, g, p.type_());
  }*/

  // We manually set up the AdamW optimizer state.
  //
  // The AdamW optimizer state consists of 3 parts:
  //
  // 1. the master copy of the parameters;
  // 2. the 1st moment moving average of the gradients; and
  // 3. the 2nd moment moving average of the gradients.
  let mut master = Vec::with_capacity(param.len());
  let mut grad_avg = Vec::with_capacity(param.len());
  let mut grad2_avg = Vec::with_capacity(param.len());
  for p in param.iter() {
    // In this example, we are using 32-bit AdamW (`AdamW32`),
    // so we up-cast the optimizer state accordingly.
    let p_ = StableCell::from(p.type_().cast(f32::dtype_()));
    let g_ = StableCell::from(p.type_().cast(f32::dtype_()));
    let g2 = StableCell::from(p.type_().cast(f32::dtype_()));
    /*println!("train:  master={:?} ty={:?} grad avg={:?} ty={:?} grad2 avg={:?} ty={:?}",
        p_, p_.type_(), g_, g_.type_(), g2, g2.type_());*/
    master.push(p_);
    grad_avg.push(g_);
    grad2_avg.push(g2);
  }

  // Each cycle corresponds to one micro-batch.
  let cycles_per_minibatch = minibatch_sz / cfg.ubat_sz as i32;
  let fin_cycle = cycles_per_minibatch * num_minibatch_iters;

  // That's enough prep. Time to train!
  for cycle_nr in 0 ..= fin_cycle {
    println!("train:  start cycle={}", cycle_nr);

    // Up to now (before the `for` loop), we have been doing
    // prepatory rituals: opening the pickle data directory,
    // loading the tokenizer, creating our language model,
    // connecting our language model parameters to tensors on
    // disk, and creating inputs.
    //
    // Now, we will begin the dataflow setup phase.
    // In `cacti`, the very first thing to do when starting
    // dataflow is to call `reset`.
    //
    // Gory details: Dataflow in `cacti` consists of an
    // interplay between a control thread (e.g. `main`) and
    // a thread-local _spine coroutine_. When you call `reset`
    // on the control thread (i.e. right now in `main`), you
    // are requesting the spine coroutine to increment its
    // internal cycle counter by one, and to reset a relevant
    // part of its internal state.
    reset();

    if cycle_nr == 0 {
      // If this is the zero-th cycle, then we need to
      // initialize the model parameters from their on-disk
      // tensors.
      for (cel, _) in inv_matches.iter() {
        // The spine coroutine maintains a _computation spine_
        // or simply the _spine_; this is analogous to the
        // "tape" or "Wengert list" if you are familiar with
        // automatic differentiation.
        //
        // In `cacti`, a "cell" (sometimes abbreviated "cel")
        // is a reference to a dataflow variable. For example,
        // the language model parameters are all cells.
        //
        // When you call dataflow functions on cells during
        // the dataflow setup phase in the control thread, you
        // are _appending dataflow instructions_ to the spine.
        // The spine coroutine itself is in a dormant state,
        // while the control thread is in control.
        //
        // Below, we call the dataflow function `mem_set_yield_`
        // on a "cel" that corresponds to a model parameter
        // matched to a pickle tensor. The name `mem_set_yield_`,
        // suggesting the reversed mnemonic "yield-set-mem",
        // can be thought of as appending the following
        // instruction sequence to the spine:
        //
        // 1. "yield": The spine coroutine should yield control
        //    back to the control thread.
        // 2. "set": "yield" will return a value from the
        //    control thread to the spine coroutine; use this value
        //    to "set" the contents of the "cel".
        // 3. "mem": the "cel" contents and the value it was
        //    "set" to should be in host/CPU memory.
        cel.mem_set_yield_();
      }

      // We call `smp_scope` to hint that the AdamW optimizer
      // state should be stored using the SMP subsystem,
      // i.e., on the host CPU memory.
      // We do this because the optimizer state is large enough
      // that it _probably_ won't fit on your GPU anyway.
      smp_scope().with(|_| {
        // Initialize the AdamW master copy of the parameters
        // with a dtype cast.
        //
        // Note that we don't have to explicitly specify the
        // dtype of the cast, because we already set the type
        // earlier (using `StableCell::from`).
        // Thus, the appropriate dtype is correctly inferred.
        for (p, p_) in param.iter().zip(master.iter()) {
          p_.set_cast(p.const_());
        }
        // Initialize the AdamW 1st moment gradient average to
        // all zeros.
        for g_ in grad_avg.iter() {
          g_.set_zeros();
        }
        // Initialize the AdamW 2nd moment gradient average to
        // all zeros.
        for g2 in grad2_avg.iter() {
          g2.set_zeros();
        }
      });

      // Initialize constants (for the positional embedding)
      model.init_constants();
    }

    // We perform an AdamW update at the beginning of the
    // cycle. This allows the condition to appear slightly
    // simpler.
    if cycle_nr != 0 && cycle_nr % cycles_per_minibatch == 0 {
      println!("train:  cycle={} adamw step...", cycle_nr);
      smp_scope().with(|_| {
        for ((((g, g_), g2), p_), p) in
            grad.iter()
            .zip(grad_avg.iter())
            .zip(grad2_avg.iter())
            .zip(master.iter())
            .zip(param.iter())
            .rev()
        {
          if p == &model.embed {
            // NB: We're supposed to zero out the zero-th row of
            // the embedding gradient, but instead we just don't
            // update the embedding at all.
            p.cache();
          } else {
            adamw.step(cycle_nr, &p_, &g_, &g2, &g);
            // We use `set_lossy_cast` here as we are downcasting
            // from the FP32 master copy to the FP16 working copy.
            p.set_lossy_cast(p_.const_());
          }
        }
      });

      // For training, it's not strictly necessary to `cache`
      // the positional embedding constants, as they remain
      // unchanged on each cycle.
      /*model.cache_constants();*/
    }

    /*// If this is the final cycle, quit while we're ahead.
    if cycle_nr == fin_cycle {
      println!("train:  fin cycle={}", cycle_nr);
      break;
    }*/

    // When the current cycle is _not_ a minibatch
    // update, we cache the gradients to enable
    // micro-batched gradient accumulation.
    //
    // (Note that if the minibatch size is the same as
    // the micro-batch size, then we _never_ accumulate
    // gradients across micro-batches.)
    if cycle_nr % cycles_per_minibatch != 0 {
      for g in grad.iter() {
        g.cache();
      }
    }

    // As we used `mem_set_yield_` to initialize the
    // model parameters, we will also use `mem_set_yield_`
    // to set the input sequence.
    in_.in_tok.mem_set_yield_();

    // We use `mem_set_yield_` to set the language modeling
    // training labels.
    in_.in_lm_tok.mem_set_yield_();

    // We use `mem_set_yield_` to set the element-wise
    // loss scaling (see below).
    in_.in_lm_loss_scale.mem_set_yield_();

    // `Llama::apply` will stage a forward pass of LLaMA;
    // i.e. it will append a bunch of dataflow instructions to
    // the spine, but they will not yet be executed, since the
    // spine coroutine is still dormant.
    let out = model.apply(in_.in_tok.const_(), in_.in_lm_tok.const_());

    // We just staged the forward pass. Now let's stage the
    // backward pass to compute the gradients for AdamW.
    //
    // The next few lines are somewhat low-level, but they
    // are all you need for automatic differentiation or
    // backprop in `cacti`.
    //
    // The underlying primitives we invoke below are:
    //
    // 1. `CellMap`: A monotone map from cells to cells.
    // 2. `CellSet`: A monotone set of cells.
    // 3. `CellMap::vadd_vjp`: A function that extends any
    //    given `CellMap` with key-value pairs corresponding
    //    to the vector-Jacobian product (a.k.a. reverse mode
    //    autodiff).
    let grad_map = CellMap::new();
    let src = CellSet::new();
    let sink = CellMap::new();
    sink.add(&out.out_lm_loss, &in_.in_lm_loss_scale);
    for (p, g) in param.iter().zip(grad.iter()) {
      src.add(p);
      grad_map.add(p, g);
    }
    grad_map.vadd_vjp(&src, &sink);

    // FP16 training can go off the rails if the gradient or
    // loss scaling is insufficient.
    //
    // To diagnose issues related to gradient or loss scaling,
    // we calculate, for each pair of parameter and gradient:
    //
    // 1. a histogram of the base-2 logarithms of the element-
    //    wise absolute values (which correctly accounts for
    //    denormalized and non-finite floating point values);
    //    and
    // 2. a count of floating point NaNs (including both
    //    signaling and non-signaling NaNs).
    //
    // The resulting output is pretty gory, and I commented
    // it out below, but it's useful to have this code around
    // when training, just in case. Anyway it's pretty cheap
    // to calculate.
    let mut param_log2_hist = Vec::with_capacity(param.len());
    let mut param_nan_count = Vec::with_capacity(param.len());
    let mut grad_log2_hist = Vec::with_capacity(param.len());
    let mut grad_nan_count = Vec::with_capacity(param.len());
    for (p, g) in param.iter().zip(grad.iter()) {
      if !(grad_map.get(p) == g) {
        println!("train:  WARNING: uh oh, param-grad mismatch!");
        panic!()
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

    if cycle_nr == 0 {
      // For debugging, let's dump some shapes and dtypes.
      println!("train:  in_lm_tok.shape={:?}", in_.in_lm_tok.shape());
      println!("train:  in_lm_tok.dtype={:?}", in_.in_lm_tok.dtype());
      println!("train:  in_lm_loss_scale.shape={:?}", in_.in_lm_loss_scale.shape());
      println!("train:  in_lm_loss_scale.dtype={:?}", in_.in_lm_loss_scale.dtype());
      println!("train:  out_lm_prob.shape={:?}", out.out_lm_prob.shape());
      println!("train:  out_lm_prob.dtype={:?}", out.out_lm_prob.dtype());
      println!("train:  out_lm_loss.shape={:?}", out.out_lm_loss.shape());
      println!("train:  out_lm_loss.dtype={:?}", out.out_lm_loss.dtype());
    }

    // At this point, we are done with dataflow setup.
    // The spine coroutine, though still in a dormant state,
    // is now flush with instructions waiting to run.
    //
    // But before we run the spine, first we call `compile`
    // to perform a static analysis on the spine. The static
    // analysis can then enable run-time optimizations to
    // reduce memory usage and avoid OOM failures.
    compile();

    // Now we are ready to run the dataflow instructions in
    // the spine.
    //
    // `resume` will pass control from the control thread
    // (i.e. `main`) to the spine coroutine. Then, the spine
    // coroutine will change state from dormant to running,
    // as it steps through the staged dataflow instructions.
    //
    // (If you are already familiar with coroutines, then
    // yes, this `resume` does what you would expect.)
    resume();

    // Earlier, we staged some `mem_set_yield_` operations
    // to the spine. As the spine coroutine runs its code,
    // it will pause when it encounters the instruction
    // for `mem_set_yield_`. Then the spine coroutine will
    // return back to the control thread (precisely, at the
    // `resume` from a few lines above).
    //
    // The spine coroutine, which is paused at a
    // `mem_set_yield_`, is now waiting to receive a value
    // from the control thread. We would now like to
    // simultaneously send a value to and resume running
    // the spine coroutine.
    //
    // To do so, for each `mem_set_yield_` above, we call a
    // corresponding `resume_put` or `resume_put_mem_with`
    // below.
    if cycle_nr == 0 {
      for (cel, _key) in inv_matches.iter() {
        let (pickty, pickfile) = pickdir.get(inv_matches.get(cel));
        resume_put(cel, &pickty, pickfile.mmap());
      }
    }
    resume_put_mem_with(&in_.in_tok, |typed_mem| {
      println!("train:  set in_tok...");
      let tok_buf = typed_mem.into_mut_slice::<u16>().unwrap();
      // Prepend 1 (= BOS token) to the input.
      tok_buf[0] = 1_u16;
      // Then, copy the tokenized text to the input.
      tok_buf[1 ..= text_tok.len()].copy_from_slice(text_tok.as_ref());
      // Last, fill the rest of the input with 0 (= pad token).
      for j in text_tok.len() + 1 .. cfg.seq_cap as _ {
        tok_buf[j] = 0;
      }
    });
    resume_put_mem_with(&in_.in_lm_tok, |typed_mem| {
      println!("train:  set in_lm_tok...");
      let tok_buf = typed_mem.into_mut_slice::<u16>().unwrap();
      // The language modeling labels are the same as
      // the input tokens, but shifted to the left by 1.
      tok_buf[ .. text_tok.len()].copy_from_slice(text_tok.as_ref());
      for i in text_tok.len() .. cfg.seq_cap as _ {
        tok_buf[i] = 0_u16;
      }
    });
    resume_put_mem_with(&in_.in_lm_loss_scale, |typed_mem| {
      println!("train:  set in_lm_loss_scale...");
      // The loss scale is proportional to the gradient
      // scaling we set near the beginning of this example,
      // and is inversely proportional to the sequence
      // length of the training example.
      //
      // We zero out the loss scaling at the 0-th position,
      // which corresponds to the BOS token. But, you could
      // choose not to do so, in which case you would need
      // to adjust the denominator of the loss scale.
      //
      // We also subtract one from the sequence length to
      // avoid training on the very last element of the
      // sequence, which by definition does not have a
      // "next token" and so has a spurious loss.
      let loss_scale = grad_scale / (text_tok.len() - 1) as f32;
      println!("train:    loss scale: {:?}", loss_scale);
      let loss_scale_buf = typed_mem.into_mut_slice::<f32>().unwrap();
      loss_scale_buf[0] = 0.0_f32;
      for i in 1 .. text_tok.len() {
        loss_scale_buf[i] = loss_scale;
      }
      for i in text_tok.len() .. cfg.seq_cap as _ {
        loss_scale_buf[i] = 0.0_f32;
      }
    });

    // Below, we read out to memory the language modeling
    // output probabilities and fine-tuning losses.
    [ out.out_lm_prob.clone(),
      out.out_lm_loss.clone(),
    ].with_mem(|typed_mems| {
      let out_lm_prob_f32 = typed_mems[0].as_slice::<f32>().unwrap();
      let out_lm_loss_f32 = typed_mems[1].as_slice::<f32>().unwrap();
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
      let ntok = cfg.tok_dim as usize;
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
      let loss_avg = loss_sum / (text_tok.len() - 1) as f32;
      println!("train:  cycle={} loss sum={:.06} avg={:.06}", cycle_nr, loss_sum, loss_avg);
    });

    // Below, we summarize the floating point digest that
    // was computed earlier by `abs_log2_hist8` and
    // `nan_count` on the model parameters.
    //
    // By default, we only display the digest if we observe
    // a floating point blowup (inf or NaN).
    //println!("train:  inspect param");
    for ((param, p_log2_hist), p_nan_count) in param.iter().zip(param_log2_hist.iter()).zip(param_nan_count.iter()) {
      [ p_log2_hist.clone(),
        p_nan_count.clone(),
      ].with_mem(|typed_mems| {
        let h = typed_mems[0].as_slice::<i64>().unwrap();
        let nan = typed_mems[1].as_slice::<i64>().unwrap();
        if !(h.len() == 0x100) {
          println!("train:  WARNING: param log2 hist: unexpected len: {} != 256", h.len());
        }
        if !(nan.len() == 1) {
          println!("train:  WARNING: param nan count: unexpected len: {} != 1", nan.len());
        }
        let nan = nan[0];
        let bias = 0x7f_u8;
        let mut total = 0;
        total += h[0];
        for &x in (&h[(bias - 24) as usize ..= (bias - 15) as usize]).iter() {
          total += x;
        }
        for &x in (&h[(bias - 14) as usize ..= (bias -  0) as usize]).iter() {
          total += x;
        }
        for &x in (&h[(bias +  1) as usize ..= (bias + 15) as usize]).iter() {
          total += x;
        }
        total += h[0xff];
        if h[0xff] != 0 {
          println!("train:  param log2 hist: zero={:?} sub={:?} -norm={:?} +norm={:?} unfin={:?} nan={:?} total={:?} label={:?}",
              h[0],
              &h[(bias - 24) as usize ..= (bias - 15) as usize],
              &h[(bias - 14) as usize ..= (bias -  0) as usize],
              &h[(bias +  1) as usize ..= (bias + 15) as usize],
              h[0xff],
              nan,
              total,
              inv_matches.get(param),
          );
          println!("train:  param log2 hist: WARNING: fp blowup: label={:?}",
              inv_matches.get(param),
          );
        }
      });
    }

    // Below, we summarize the floating point digest that
    // was computed earlier by `abs_log2_hist8` and
    // `nan_count` on the gradients.
    //
    // By default, we only display the digest if we observe
    // a floating point blowup (inf or NaN).
    //println!("train:  inspect gradient");
    for (((p, g), g_log2_hist), g_nan_count) in param.iter().zip(grad.iter()).zip(grad_log2_hist.iter()).zip(grad_nan_count.iter()) {
      if !(grad_map.get(p) == g) {
        println!("train:  WARNING: grad mismatch");
        continue;
      }
      [ g_log2_hist.clone(),
        g_nan_count.clone(),
      ].with_mem(|typed_mems| {
        let h = typed_mems[0].as_slice::<i64>().unwrap();
        let nan = typed_mems[1].as_slice::<i64>().unwrap();
        if !(h.len() == 0x100) {
          println!("train:  WARNING: grad log2 hist: unexpected len: {} != 256", h.len());
        }
        if !(nan.len() == 1) {
          println!("train:  WARNING: grad nan count: unexpected len: {} != 1", nan.len());
        }
        let nan = nan[0];
        let bias = 0x7f_u8;
        let mut total = 0;
        total += h[0];
        for &x in (&h[(bias - 24) as usize ..= (bias - 15) as usize]).iter() {
          total += x;
        }
        for &x in (&h[(bias - 14) as usize ..= (bias -  0) as usize]).iter() {
          total += x;
        }
        for &x in (&h[(bias +  1) as usize ..= (bias + 15) as usize]).iter() {
          total += x;
        }
        total += h[0xff];
        if h[0xff] != 0 {
          println!("train:  grad log2 hist: zero={:?} sub={:?} -norm={:?} +norm={:?} unfin={:?} nan={:?} total={:?} label={:?}",
              h[0],
              &h[(bias - 24) as usize ..= (bias - 15) as usize],
              &h[(bias - 14) as usize ..= (bias -  0) as usize],
              &h[(bias +  1) as usize ..= (bias + 15) as usize],
              h[0xff],
              nan,
              total,
              inv_matches.get(p),
          );
          println!("train:  param log2 hist: WARNING: fp blowup: label={:?}",
              inv_matches.get(p),
          );
        }
      });
    }

    println!("train:  end cycle={}", cycle_nr);
  }

  // And that's it for this example.
  // Thank you for reaching the end!
  println!("train:  done");
}
