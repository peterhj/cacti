extern crate cacti;

use cacti::*;

fn main() {
  for k in 0 ..= 10 {
    reset();
    //let x = StableCell::array([1], "f32");
    let x = StableCell::from(3.14_f32).snapshot();
    //let x = StableCell::scalar(3.14_f32 * (k as f32));
    //x.cache();
    println!("boot: x={:?}", x);
    let y = StableCell::from(2.71828_f32);
    println!("boot: y={:?}", y);
    let z = StableCell::from(0.577_f32);
    println!("boot: z={:?}", z);
    compile();
    resume();
  }
  println!("boot: done");
}
