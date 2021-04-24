use std::{cmp, fmt, sync::Arc};

pub type Obj = Arc<ObjType>;

#[derive(Clone, Copy)]
pub enum ObjType {
    Nil,
    Number(Number),
    Pair(Obj, Obj),
    Proc(Box<dyn Fn(Obj) -> Obj>),
    String_(String),
    Symbol(String),
    Boolean(bool),
}

impl fmt::Debug for ObjType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ObjType::*;
        let txt = match self {
            Nil => "Obj::Nil".to_owned(),
            Pair(x, y) => format!("Obj::Pair({:?}, {:?})", x, y),
            Proc(_) => "Obj::Proc".to_owned(),
            Number(n) => format!("{:?}", n),
            String_(text) => format!("Obj::String({:?})", n),
            Symbol(s) => format!("Obj::Symbol({:?})", s),
            Boolean(t) => if t { "#true" } else { "#false" },
        };
        f.write_str(&txt)
    }
}

impl PartialEq for ObjType {
    fn eq(&self, other: &Self) -> bool {
        use ObjType::*;
        match (self, other) {
            (Nil, Nil) => true,
            (Number(n1), Number(n2)) => n1 == n2,
            (Pair(o11, o12), Pair(o21, o22)) => o11 == o21 && o12 == o22,
            (String_(s1), String_(s2)) => s1 == s2,
            (Symbol(s1), Symbol(s2)) => s1 == s2,
            (Boolean(b1), Boolean(b2)) => b1 == b2,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Number {
    Complex(f64, f64),
    Real(f64),
    Rational(i64, u64),
    Integral(i64),
}

impl From<Number> for f64 {
    fn from(number: Number) -> Self {
        match number {
            Complex(r, i) => (r * r + i * i).sqrt()
                * if r >= 0 { 1.0 } else { -1.0 }
                * if i >= 0 { 1.0 } else { -1.0 },
            Real(r) => r,
            Rational(n, d) => n as f64 / d as f64,
            Integral(i) => i as f64,
        }
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Number {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        let me = f64::from(self);
        let you = f64::from(other);
        if me > you {
            cmp::Ordering::Greater
        } else if me < you {
            cmp::Ordering::Less
        } else {
            cmp::Ordering::Equal
        }
    }
}

impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Number::*;
        let txt = match self {
            Complex(_, _) => format!("Number::Complex({})", &self),
            Real(_) => format!("Number::Real({})", &self),
            Rational(_, _) => format!("Number::Rational({})", &self),
            Integral(_) => format!("Number::Integral({})", &self),
        };
        f.write_str(&txt)
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Number::*;
        let txt = match self {
            Complex(r, i) => format!("{:+}{:+}i", r, i),
            Real(r) => format!("{}", r),
            Rational(n, d) => format!("{}/{}", n, d),
            Integral(i) => format!("{}", i),
        };
        f.write_str(&txt)
    }
}
