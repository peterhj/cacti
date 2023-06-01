extern crate cacti;

use cacti::*;

fn main() {
  //cacti::ctx::ctx_cfg_set_default_compute(cacti::cell::PMachSpec::Smp);
  for k in 0 ..= 10 {
    reset();
    //let x = StableCell::array([1], "f32");
    let x = StableCell::from(3.14_f32);
    //let x = StableCell::scalar(3.14_f32 * (k as f32));
    //x.cache();
    //let y = x + 2.78_f32;
    compile();
    resume();
  }
  println!("boot: done");
}
