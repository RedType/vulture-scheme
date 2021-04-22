use std::{fmt, sync::Arc};

pub type Obj = Arc<ObjType>;

pub enum ObjType {
    Atom(String),
    Nil,
    Pair(Obj, Obj),
    Proc(Box<dyn Fn(Obj) -> Obj>),
}

impl fmt::Debug for ObjType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ObjType::*;
        let txt = match self {
            Atom(s) => format!("Obj::Atom({:?})", s),
            Nil => "Obj::Nil".to_owned(),
            Pair(x, y) => format!("Obj::Pair({:?}, {:?})", x, y),
            Proc(_) => "Obj::Proc".to_owned(),
        };
        f.write_str(&txt)
    }
}

impl PartialEq for ObjType {
    fn eq(&self, other: &Self) -> bool {
        use ObjType::*;
        match (self, other) {
            (Atom(s1), Atom(s2)) => s1 == s2,
            (Nil, Nil) => true,
            (Pair(o11, o12), Pair(o21, o22)) => o11 == o21 && o12 == o22,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Number {
    Complex(f64, f64),
    Real(f64),
    Rational(i64, u64),
    Integral(i64),
}
