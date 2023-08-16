extern crate cacti;

use cacti::prelude::*;
use cacti::algo::str::{safe_ascii};
use cacti::librarium::llama::{LlamaConfig, LlamaCached};
use cacti::librarium::sentencepiece::{SentencePieceTokenizer};
use cacti::util::pickle::{PickleDir};

use std::io::{Write, stdout};

fn main() {
  // To run this inference example, you will need to have
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
  // You will need to increase this if you would like a longer
  // prompt or completion.
  cfg.seq_cap = 256;

  println!("deploy: llama config: {:?}", cfg);

  // The directory we provided at the beginning of `main`
  // should contain "pytorch_model*.bin" pickle files.
  // Opening a `PickleDir` will safely parse the files'
  // metadata, without loading all the data just yet.
  let pickdir = PickleDir::open(data_dir).unwrap();
  println!("deploy: loaded pickle dir");

  // Also load a SentencePiece tokenizer from the
  // "tokenizer.model" file in the above directory.
  let tokenizer = SentencePieceTokenizer::from_dir(data_dir).unwrap();
  println!("deploy: loaded sentencepiece tokenizer");
  println!("deploy: tokenizer: n={:?}", tokenizer.num_pieces());
  println!("deploy: tokenizer: unk={:?}", tokenizer.unk_id());
  println!("deploy: tokenizer: bos={:?}", tokenizer.bos_id());
  println!("deploy: tokenizer: eos={:?}", tokenizer.eos_id());
  println!("deploy: tokenizer: pad={:?}", tokenizer.pad_id());

  // And this will be our toy example for inference.
  // A real classic. (A 19th century translation by
  // Richard Crawley, via Project Gutenberg.)
  let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its grounds. The preparations of both the combatants were in every department in the last state of perfection; and he could see the rest of the Hellenic race taking sides in the quarrel; those who delayed doing so at once having it in contemplation. Indeed this was the greatest movement yet known in history, not only of the";
  //let text_str = "Thucydides, an Athenian, wrote the history of the war between the Peloponnesians and the Athenians, beginning at the moment that it broke out, and believing that it would be a great war and more worthy of relation than any that had preceded it. This belief was not without its grounds. The preparations of both the combatants were in every department in the last state of perfection; and he could see the rest of the Hellenic race taking sides in the quarrel; those who delayed doing so at once having it in contemplation. Indeed this was the greatest movement yet known in history, not only of the Hellenes, but of a large part of the barbarian world-- I had almost said of mankind. For though the events of remote antiquity, and even those that more immediately preceded the war, could not from lapse of time be clearly ascertained, yet the evidences which an inquiry carried as far back as was practicable leads me to trust, all point to the conclusion that there was nothing on a great scale, either in war or in other matters.";
  println!("deploy: data: text str=\"{}\"", text_str);
  println!("deploy: data: text str.len={}", text_str.len());

  // Now let's tokenize that string!
  // We should get roughly 130 16-bit tokens.
  let text_tok = tokenizer.encode16(text_str).unwrap();
  println!("deploy: data: text tok={:?}", text_tok.as_ref());
  println!("deploy: data: text tok.len={}", text_tok.len());

  // `LlamaCached::from` will create a LLaMA model from the
  // provided config that is suitable for inference.
  //
  // `LlamaCached` is implemented in (src/librarium/llama.rs),
  // which you may also want to read.
  let mut model = LlamaCached::from(cfg);

  // So, we have an in-memory `model`, and we have a
  // `pickdir` that represents the contents on disk.
  // We will need to connect the two in order to figure out
  // which on-disk pickle tensors map to which in-memory
  // cells and vice versa. To do so, we use a convenience
  // function, `LlamaCached::match_pickle_dir`, which itself
  // is implemented using parts from (src/util/cell.rs) and
  // (src/util/safepickle.rs).
  let inv_matches = model.match_pickle_dir(&pickdir);
  for (cel, key) in inv_matches.iter() {
    println!("deploy: matches: key={:?} cel={:?}", key, cel);
  }

  // Create a fresh `Vec` of language model input cells,
  // where the length of the `Vec` is the micro-batch size.
  // The full type of `in_` is `Vec<LanguageModelDeployInput>`;
  // see (src/librarium/lm.rs).
  let mut in_ = model.fresh_input();

  // We'll run this example for 100 cycles.
  for cycle_nr in 0 .. 100 {
    //println!("deploy: start cycle={}", cycle_nr);

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
      for (cel, _key) in inv_matches.iter() {
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

      // Initialize constants (for the positional embedding)
      model.init_constants();

      // Initialize the cached KV-activations to zeros.
      model.init_state();
    } else {
      // For inference, we don't need to `cache` the model parameters,
      // or the positional embedding constants. So long as they are not
      // otherwise updated, they will remain the same on each cycle.
      /*for p in param.iter() {
        p.cache();
      }
      model.cache_constants();*/

      // On the other hand, we _do_ need to `cache` the previous cycle's
      // KV activations, as we want to update them during this cycle's
      // forward pass while preserving their values computed during the
      // previous cycle.
      model.cache_state();
    }

    if cycle_nr == 0 {
      for i in 0 .. cfg.ubat_sz as usize {
        // As we used `mem_set_yield_` to initialize the
        // model parameters, we will also use `mem_set_yield_`
        // to set the input sequence.
        in_[i].in_tok.mem_set_yield_();
      }
    } else {
      for i in 0 .. cfg.ubat_sz as usize {
        // The inference model is autoregressive, and so will
        // automatically update the input sequence with freshly
        // predicted tokens (up to the maximum sequence length).
        // Thus, after the zero-th cycle, we just need to cache
        // the input sequence.
        in_[i].in_tok.cache();
      }
    }

    // `LlamaCached::apply` will stage a forward pass of LLaMA;
    // i.e. it will append a bunch of dataflow instructions to
    // the spine, but they will not yet be executed, since the
    // spine coroutine is still dormant.
    //
    // Note that we add 1 to the sequence length (in tokens)
    // since we have to prepend the BOS token.
    let out = if cycle_nr == 0 {
      model.apply(&mut in_, 0, 1 + text_tok.len() as i64)
    } else {
      model.apply(&mut in_, 1 + text_tok.len() as i64 + cycle_nr - 1, 1 + text_tok.len() as i64 + cycle_nr)
    };

    if cycle_nr == 0 {
      // For debugging, let's dump some shapes and dtypes.
      println!("deploy: in_tok.shape={:?}", in_[0].in_tok.shape());
      println!("deploy: in_tok.dtype={:?}", in_[0].in_tok.dtype());
      println!("deploy: out_lm_logit.shape={:?}", out[0].out_lm_logit.shape());
      println!("deploy: out_lm_logit.dtype={:?}", out[0].out_lm_logit.dtype());
      println!("deploy: out_lm_prob.shape={:?}", out[0].out_lm_prob.shape());
      println!("deploy: out_lm_prob.dtype={:?}", out[0].out_lm_prob.dtype());
      println!("deploy: out_lm_tok.shape={:?}", out[0].out_lm_tok.shape());
      println!("deploy: out_lm_tok.dtype={:?}", out[0].out_lm_tok.dtype());
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

    if cycle_nr == 0 {
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
      for (cel, _) in inv_matches.iter() {
        let (pickty, pickfile) = pickdir.get(inv_matches.get(cel));
        resume_put(cel, &pickty, pickfile.mmap());
      }
      resume_put_mem_with(&in_[0].in_tok, |typed_mem| {
        println!("deploy: set in_tok...");
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
    }

    // Below, we read out the input and output tokens from
    // running the language model, and display them in the
    // format of "Prompt: ... Completion: ...".
    [ in_[0].in_tok.clone(),
      out[0].out_lm_tok.clone(),
    ].with_mem(|typed_mems| {
      let in_tok_u16 = typed_mems[0].as_slice::<u16>().unwrap();
      let out_tok_u16 = typed_mems[1].as_slice::<u16>().unwrap();
      let (start_pos, fin_pos) = if cycle_nr == 0 {
        (1, 1 + text_tok.len())
      } else {
        (1 + text_tok.len() + (cycle_nr) as usize, 1 + text_tok.len() + (cycle_nr) as usize)
      };
      for pos in start_pos ..= fin_pos {
        let prev_pos = pos - 1;
        let act_prev_tok = if prev_pos >= 1 && prev_pos < text_tok.len() + 1 {
          text_tok.as_ref()[prev_pos - 1]
        } else {
          0
        };
        let prev_tok = in_tok_u16[prev_pos];
        let next_tok = out_tok_u16[pos - start_pos];
        let auto_next_tok = in_tok_u16[pos];
        if pos >= 1 + text_tok.len() {
          assert_eq!(next_tok, auto_next_tok);
        }
        if pos <= 1 + text_tok.len() {
          if pos == 1 {
            println!();
            println!("Prompt:");
          }
          let s = tokenizer.id_to_piece(prev_tok as _).unwrap();
          for c in s.chars() {
            if c as u32 == 9601 {
              print!(" ");
            } else {
              print!("{}", c);
            }
          }
        }
        if pos >= 1 + text_tok.len() {
          if pos == 1 + text_tok.len() {
            println!();
            println!();
            println!("Completion:");
          }
          let s = tokenizer.id_to_piece(next_tok as _).unwrap();
          for c in s.chars() {
            if c as u32 == 9601 {
              print!(" ");
            } else {
              print!("{}", c);
            }
          }
        }
        // Flush stdout to prevent the appearance of
        // batching.
        stdout().lock().flush().unwrap();
      }
    });

    //println!("deploy: end cycle={}", cycle_nr);
  }
  println!();
  println!();

  // And that's it for this example.
  // Thank you for reaching the end!
  println!("deploy: done");
}
