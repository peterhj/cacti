extern crate cacti;

use cacti::*;

fn main() {
  //cacti::ctx::ctx_cfg_set_default_compute(cacti::cell::PMachSpec::Smp);
  reset();
  let x = StableCell::new_scalar(3.14_f32);
  //let x = StableCell::new_array([1], "f32");
  //x.cache();
  compile();
  resume();
}
