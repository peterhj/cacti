fn emit_index(nd: i8, curs: i8, buf: &mut Vec<&str>) {
  for &tok in ["IRange", "IRangeTo", "IRangeFrom", "IRangeFull"].iter() {
    buf.push(tok);
    if curs + 1 > nd {
      unreachable!();
    } else if curs + 1 == nd {
      let mut full = true;
      for d in 0 .. nd {
        if buf[d as usize] != "IRangeFull" {
          full = false;
          break;
        }
      }
      if full {
        print!("ifull!({}, (", nd);
      } else {
        print!("index!({}, (", nd);
      }
      print!("{}", buf[0]);
      for d in 1 .. nd {
        print!(", {}", buf[d as usize]);
      }
      println!("));");
    } else {
      emit_index(nd, curs + 1, buf);
    }
    buf.pop();
  }
}

fn emit_iproj(nd: i8, curs: i8, buf: &mut Vec<&str>, prev_mat: bool) {
  for &tok in ["i64", "IRange", "IRangeTo", "IRangeFrom", "IRangeFull"].iter() {
    let mut mat = prev_mat;
    if tok == "i64" {
      mat = true;
    }
    buf.push(tok);
    if curs + 1 > nd {
      unreachable!();
    } else if curs + 1 == nd {
      if mat {
        print!("iproj!({}, (", nd);
        print!("{}", buf[0]);
        for d in 1 .. nd {
          print!(", {}", buf[d as usize]);
        }
        println!("));");
      }
    } else {
      emit_iproj(nd, curs + 1, buf, mat);
    }
    buf.pop();
  }
}

fn main() {
  for nd in 1 ..= 4 {
    emit_index(nd, 0, &mut Vec::new());
    emit_iproj(nd, 0, &mut Vec::new(), false);
  }
}
