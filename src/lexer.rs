use crate::{
    error::Error,
    types as ty,
};
use lazy_static::lazy_static;
use regex::{Regex, RegexSet};

#[derive(Clone, Debug, PartialEq)]
pub enum Lexeme {
    Identifier(String),
    LParen, RParen,
    Num(ty::Number),
    String_(String),
    True, False,
}

#[derive(Debug)]
pub enum Fragment {
    Nothing,
    BadString(Error),
    Lex(Lexeme),
    Imag(f64),
    Real(f64),
}

pub fn lex<'a>(corpus: &'a str) -> Result<Vec<Lexeme>, Error> {
    let mut frags: Vec<Fragment> = Vec::new();
    let mut head = corpus;
    while head.len() > 0 {
        let (frag, offset) = find_next(head)?;
        let offset = if offset > head.len() { head.len() } else { offset };
        head = &head[offset..];
        frags.push(frag);
    }
    let lexemes = process_fragments(frags)?;
    Ok(lexemes)
}

fn find_next(corpus: &str) -> Result<(Fragment, usize), Error> {
    lazy_static! {
        static ref NEWLINE: Regex = Regex::new(r"[\r\n]").unwrap();
    }
    match &ALL.matches(corpus).iter().collect::<Vec<_>>()[..] {
        [i] => LEXERS[*i](corpus).ok_or_else(|| panic!("Lexer mismatch")),
        a @ [_, ..] => {
            let next_newline = match NEWLINE.find(corpus) {
                Some(m) => ..m.start(),
                None => ..corpus.len(),
            };
            let found_frags =
                a.iter().map(|i| LEXERS[*i](corpus)
                                   .unwrap_or_else(|| panic!("Lexer mismatch")).0);
            Err(Error::Ambiguous(corpus[next_newline].to_owned(), found_frags.collect()))
        },
        [] => {
            let next_newline = match NEWLINE.find(corpus) {
                Some(m) => ..m.start(),
                None => ..corpus.len(),
            };
            Err(Error::NoMatch(corpus[next_newline].to_owned()))
        },
    }
}

fn process_fragments(fragments: Vec<Fragment>) -> Result<Vec<Lexeme>, Error> {
    let mut frags = fragments.into_iter().peekable();
    let mut lexemes = Vec::new();
    let mut skip_next = false;
    while let Some(frag) = frags.next() {
        if skip_next {
            skip_next = false;
            continue;
        }
        match frag {
            Fragment::Nothing => (),
            Fragment::Lex(l) => lexemes.push(l),
            Fragment::Real(n) =>
                if let Some(Fragment::Imag(m)) = frags.peek() {
                    skip_next = true;
                    let complex = Lexeme::Num(ty::Number::Complex(n, *m));
                    lexemes.push(complex);
                } else {
                    let real = Lexeme::Num(ty::Number::Real(n));
                    lexemes.push(real);
                },
            Fragment::Imag(n) => lexemes.push(Lexeme::Num(ty::Number::Complex(0.0, n))),
            Fragment::BadString(e) => return Err(e),
        }
    }
    Ok(lexemes)
}

type Lexer = dyn for<'a> Fn(&'a str) -> Option<(Fragment, usize)> + Sync;
macro_rules! gen_matchers {
    { $($f:ident => $re:expr),+$(,)? } => {
        lazy_static! {
            static ref ALL: RegexSet = RegexSet::new(&[$($re),+]).unwrap();
            static ref LEXERS: Vec<Box<Lexer>> =
                vec![$(Box::new(|c| {
                    let re = Regex::new($re).unwrap();
                    decode::$f(re, c)
                })),+];
        }
    };
}

gen_matchers! {
    parens => r"[()]",
    true_ => r"^#t(?:rue)?",
    false_ => r"^#f(?:alse)?",
    string => r#"^"((?:[^"\\]|\\"|\\\\|\\\s*)*)""#,
    inf_nan => r"^(\+|-)(inf|nan)\.0",
    imaginary_number => r"(?x)^
        ([\+-]?\d*\.?\d+ # coefficient
        (e[\+-]?\d+)?)?i # exponent (and i)",
    real_number => r"(?x)^
        ([\+-]?\d*\.\d+) # number
        (e[\+-]?\d+)?    # exponent",
    rational_number => r"(?x)^
        ([\+-]?\d+)/(\d+) # numerator / denominator
        (e[\+-]?\d+)?     # exponent",
    integer => r"(?x)^
        ([\+-]?\d+)   # number
        (e[\+-]?\d+)? # exponent",
    identifier => r"^\p{Alphabetic}\w*",
    whitespace => r"^\s+",
    line_comment => r"^;[^\r\n]*\r|\n|(?:\r\n)",
}

mod decode {
    use crate::{
        error::Error,
        types::Number as N,
        lexer::{
            Lexeme as L,
            Fragment as F
        },
    };
    use regex::Regex;
    use std::str::FromStr;
    use unicode_segmentation::UnicodeSegmentation;

    fn str_is_whitespace(s: &str) -> bool {
        s.chars().fold(true, |a, c| a && c.is_whitespace())
    }

    fn replace_escapes(s: &str) -> Result<String, Error> {
        let mut gs = s.graphemes(true).peekable();
        let mut string = String::new();
        while let Some(g) = gs.next() {
            match g {
                r"\" => match gs.next() {
                    None => return Err(Error::EscapeAtEndOfString(s.to_owned())),
                    Some(r"\") => string.push('\\'),
                    Some("n") => string.push_str("\n"),
                    Some("r") => string.push('\r'),
                    Some("t") => string.push('\t'),
                    Some("0") => string.push('\0'),
                    Some("\"") => string.push('"'),
                    // consume whitespace after a \
                    Some(g) if str_is_whitespace(g) => {
                        while let Some(g) = gs.peek() {
                            if str_is_whitespace(g) {
                                let _ = gs.next();
                            }
                        }
                    },
                    Some(g) => return Err(Error::InvalidEscapeSequence(format!("\\{}", g))),
                },
                g => string.push_str(g),
            }
        }
        Ok(string)
    }

    fn lex(le: L) -> F { F::Lex(le) }

    fn do_exp_int(s: &str, x: i64) -> i64 {
        let exp_str = s.chars().skip(1).collect::<String>(); // toss the e
        let exp = i64::from_str(&exp_str).unwrap();
        let (inv, exp) = if exp < 0 { (true, -exp) } else { (false, exp) };
        (0..exp).fold(x, |a, _| if inv { (a / 10) as i64 } else { a * 10 })
    }

    fn do_exp_float(s: &str, x: f64) -> f64 {
        let exp_str = s.chars().skip(1).collect::<String>(); // toss the e
        let exp = i64::from_str(&exp_str).unwrap();
        let (inv, exp) = if exp < 0 { (true, -exp) } else { (false, exp) };
        (0..exp).fold(x, |a, _| if inv { a / 10.0 } else { a * 10.0 })
    }

    pub fn parens(re: Regex, corpus: &str) -> Option<(F, usize)> {
        match re.find(corpus).map(|m| m.as_str()) {
            Some("(") => Some((lex(L::LParen), 1)),
            Some(")") => Some((lex(L::RParen), 1)),
            None => None,
            _ => panic!("Parens false positive"),
        }
    }

    pub fn true_(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.find(corpus).map(|m| (lex(L::True), m.end()))
    }

    pub fn false_(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.find(corpus).map(|m| (lex(L::False), m.end()))
    }

    pub fn string(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.captures(corpus)
          .map(|cs| match (cs.get(0), cs.get(1)) {
              (Some(m0), Some(m1)) => {
                let text = replace_escapes(m1.as_str());
                match text {
                    Ok(t) => Some((lex(L::String_(t)), m0.end())),
                    Err(e) => Some((F::BadString(e), m0.end())),
                }
              },
              _ => None,
          })
          .flatten()
    }

    pub fn imaginary_number(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.captures(corpus)
          .map(|cs| {
              let coeff = match cs.get(1) {
                  Some(m) => f64::from_str(m.as_str()).expect("Imaginary false positive"),
                  None => 1.0,
              };
              let result = match cs.get(2) {
                  Some(exp) => do_exp_float(exp.as_str(), coeff),
                  None => coeff,
              };
              cs.get(0).map(|m| (F::Imag(result), m.end()))
          })
          .flatten()
    }

    pub fn inf_nan(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.captures(corpus)
          .map(|cs| match (cs.get(0), cs.get(1), cs.get(2)) {
              (Some(m0), Some(m1), Some(m2)) => {
                  let num = match (m1.as_str(), m2.as_str()) {
                      (_, "nan") => f64::NAN,
                      ("+", "inf") => f64::INFINITY,
                      ("-", "inf") => f64::NEG_INFINITY,
                      _ => panic!("Inf nan captured wrongly"),
                  };
                  Some((lex(L::Num(N::Real(num))), m0.end()))
              },
              _ => None,
          })
          .flatten()
    }

    pub fn real_number(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.find(corpus).map(|m| (
            F::Real(f64::from_str(m.as_str()).expect("Real false positive")),
            m.end()
        ))
    }

    pub fn rational_number(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.captures(corpus)
          .map(|cs| {
              let num = match (cs.get(1), cs.get(2), cs.get(3)) {
                  (Some(numerator), Some(denominator), exp) => {
                      let a = i64::from_str(numerator.as_str())
                                .expect("Rational false positive");
                      let b = u64::from_str(denominator.as_str())
                                .expect("Rational false positive");
                      let a_e = if let Some(e) = exp { do_exp_int(e.as_str(), a) } else { a };

                      lex(L::Num(N::Rational(a_e, b)))
                  },
                  _ => panic!("Rational captured wrongly")
              };
              if let Some(m) = cs.get(0) {
                  Some((num, m.end()))
              } else {
                  panic!("Rational captured wrongly");
              }
          })
          .flatten()
    }

    pub fn integer(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.captures(corpus)
          .map(|cs| {
              let num = match (cs.get(1), cs.get(2)) {
                  (Some(int), exp) => {
                      let n = i64::from_str(int.as_str())
                                .expect("Integer false positive");
                      let result = if let Some(e) = exp { do_exp_int(e.as_str(), n) } else { n };
                      lex(L::Num(N::Integer(result)))
                  },
                  _ => panic!("Integer captured wrongly")
              };
              if let Some(m) = cs.get(0) {
                  Some((num, m.end()))
              } else {
                  panic!("Integer captured wrongly");
              }
          })
          .flatten()
    }

    pub fn identifier(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.find(corpus)
          .map(|m| (
              lex(L::Identifier(corpus[..m.end()].to_owned())),
              m.end()
          ))
    }

    pub fn whitespace(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.find(corpus).map(|m| (F::Nothing, m.end()))
    }

    pub fn line_comment(re: Regex, corpus: &str) -> Option<(F, usize)> {
        re.find(corpus).map(|m| (F::Nothing, m.end()))
    }
}

#[test]
fn lexeme_validation() {
    use Lexeme::*;
    let first = |v: Result<Vec<Lexeme>, _>| v.unwrap()[0].clone();
    assert_eq!(first(lex("#t")), True);
    assert_eq!(first(lex("#true")), True);
    assert_eq!(first(lex("#f")), False);
    assert_eq!(first(lex("#false")), False);
    assert_eq!(first(lex(r#""hello world""#)), String_("hello world".to_owned()));
    assert_eq!(first(lex(r#""\"hello\" \"world\"""#)), String_("\"hello\" \"world\"".to_owned()));
    assert_eq!(first(lex("(")), LParen);
    assert_eq!(first(lex(")")), RParen);
}
