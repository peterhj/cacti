extern crate cacti;

use cacti::*;

fn main() {
  let x_ = StableCell::new();
  println!("boot: x_={:?}", x_);
  let y_ = StableCell::new();
  println!("boot: y_={:?}", y_);
  for k in 0 ..= 10 {
    reset();
    //let x = StableCell::array([1], "f32");
    let x = StableCell::from(3.14_f32).snapshot();
    //let x = StableCell::set_scalar(3.14_f32 * (k as f32));
    //x.cache();
    println!("boot: x={:?}", x);
    let y = StableCell::from(2.71828_f32);
    println!("boot: y={:?}", y);
    let z = StableCell::from(0.577_f32);
    println!("boot: z={:?}", z);
    let w = StableCell::scalar("f32");
    //w.mem_set_yield_();
    compile();
    let ret = resume();
    println!("boot: ret={:?} (1)", ret);
    /*let ret = resume_put_mem_val(w, &0.38_f32);
    //let ret = resume_put_mem_fun(w, |ty, mreg| { ... });
    println!("boot: ret={:?} (2)", ret);*/
  }
  println!("boot: done");
}
