use crate::{error::LexError as Error, types::Number};
use once_cell::sync::Lazy;
use regex::{Regex, RegexSet};
use unicode_segmentation::UnicodeSegmentation;

pub type Token = Tagged<Lexeme>;

#[derive(Debug)]
pub struct Tagged<T> {
    pub elem: T,
    pub line: usize,
    pub col: usize,
    pub bytes: usize, // number of bytes that this lexeme took up in the corpus
}

#[derive(Clone, Debug, PartialEq)]
pub enum Lexeme {
    Identifier(String),
    LParen,
    RParen,
    Num(Number),
    String_(String),
    True,
    False,
}

#[derive(Debug)]
pub struct Corpus<'a> {
    pub text: &'a str,
    line: usize,
    col: usize,
}

impl<'a> Corpus<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            line: 1,
            col: 1,
        }
    }

    pub fn advance_by(&mut self, bytes: usize) {
        let skipped_lines = &self.text[..bytes].split('\n');
        let lines = skipped_lines.clone().count() - 1;
        let cols = skipped_lines
            .clone()
            .last()
            .unwrap_or("")
            .graphemes(true)
            .count();
        self.text = &self.text[bytes..];
        self.line += lines;
        self.col = if lines == 0 {
            self.col + cols
        } else {
            cols + 1
        };
    }

    pub fn tag<T>(&self, elem: T, bytes: usize) -> Tagged<T> {
        Tagged {
            bytes,
            elem,
            line: self.line,
            col: self.col,
        }
    }
}

pub fn lex(corpus: &str) -> Result<Vec<Token>, Error> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut head = Corpus::new(corpus);
    while !head.text.is_empty() {
        // lexers are listed in order of priority (low = better)
        let lexer_idxs = ALL.matches(head.text);
        let token_res = match lexer_idxs.into_iter().min() {
            Some(i) => LEXERS[i](&head),
            None => return Err(Error::NoMatch(head.text.to_owned())),
        };
        match token_res {
            Ok(t) => {
                head.advance_by(t.bytes);
                tokens.push(t);
            }
            Err(Error::NoToken(bytes)) => head.advance_by(bytes),
            Err(other_error) => return Err(other_error),
        };
    }
    Ok(tokens)
}

type Lexer = dyn for<'a> Fn(&'a Corpus) -> Result<Token, Error> + Send + Sync;
macro_rules! gen_matchers {
    { $($f:ident => $re:expr),+$(,)? } => {
        static ALL: Lazy<RegexSet> = Lazy::new(||
            RegexSet::new(&[$($re),+]).unwrap()
        );
        static LEXERS: Lazy<Vec<Box<Lexer>>> = Lazy::new(||
            vec![$(Box::new(|c| {
                let re = Regex::new($re).unwrap();
                decode::$f(re, c)
            })),+]
        );
    };
}

gen_matchers! {
    parens => r"^[\(\)]",
    true_ => r"^#t(?:rue)?",
    false_ => r"^#f(?:alse)?",
    string => r#"(?x)^
        "                 # open quote
            ((?:          # match one of:
                [^"\\]  | #   any character that isnt " or \
                \\[^\s] | #   "\C" where C is any non whitespace
                \\\s*     #   "\(S*)" where S is any whitespace
            )*)           # 0 or more times
        "                 # close quote
    "#,
    inf_nan => r"^(\+|-)(inf|nan)\.0",
    rational_number => r"(?x)^
        ([\+-]?\d+       # numerator
        (?:e[\+-]?\d+)?) # numerator exponent
        /                # divided by
        ([\+-]?\d+       # denominator
        (?:e[\+-]?\d+)?) # denominator exponent",
    complex_number => r"(?x)^
        ([\+-]?\d*\.?\d+  # real part
        (?:e[\+-]?\d+)?)? # real exponent
        ([\+-]?\d*\.?\d+  # imaginary part
        (?:e[\+-]?\d+)?)i # imaginary exponent (and i)",
    real_number => r"(?x)^
        ([\+-]?\d*\.\d+) # real
        (e[\+-]?\d+)?    # exponent",
    integer => r"(?x)^
        [\+-]?\d+       # integer
        (?:e[\+-]?\d+)? # exponent",
    identifier => r"^[\w\pP\pS--\d][\w\pP\pS]*",
    whitespace => r"^\s+",
    line_comment => r"^;[^\r\n]*\r|\n|(?:\r\n)",
}

mod decode {
    use crate::{
        error::LexError as Error,
        lexer::{Corpus, Lexeme, Token},
        types::Number,
    };
    use regex::Regex;
    use std::str::FromStr;
    use unicode_segmentation::UnicodeSegmentation;

    type Result<T> = std::result::Result<T, Error>;

    fn str_is_whitespace(s: &str) -> bool {
        s.chars().all(char::is_whitespace)
    }

    fn replace_escapes(s: &str) -> Result<String> {
        let mut gs = s.graphemes(true).peekable();
        let mut string = String::new();
        while let Some(g) = gs.next() {
            match g {
                r"\" => match gs.next() {
                    None => return Err(Error::EscapeAtEndOfString(s.to_owned())),
                    Some(r"\") => string.push('\\'),
                    Some("n") => string.push('\n'),
                    Some("r") => string.push('\r'),
                    Some("t") => string.push('\t'),
                    Some("0") => string.push('\0'),
                    Some("\"") => string.push('"'),
                    // consume whitespace after a \
                    Some(g) if str_is_whitespace(g) => {
                        while let Some(g) = gs.peek() {
                            if str_is_whitespace(g) {
                                let _ = gs.next();
                            } else {
                                break;
                            }
                        }
                        string.push(' ');
                    }
                    Some(g) => return Err(Error::InvalidEscapeSequence(format!(r"\{}", g))),
                },
                g => string.push_str(g),
            }
        }
        Ok(string)
    }

    fn parse_int(full_int_str: &str) -> i64 {
        let mut split = full_int_str.split('e');
        let int_str = split.next().expect("Integer false positive");
        let exp_str = split.next(); // may not be present
        let base_int = i64::from_str(int_str).expect("Integer false positive");
        let base_exp = exp_str
            .map(|s| i64::from_str(s).expect("Integer exponent false positive"))
            .unwrap_or(0);
        let (inverse, exp) = if base_exp >= 0 {
            (false, base_exp)
        } else {
            (true, -base_exp)
        };
        (0..exp).fold(base_int, |a, _| if inverse { a / 10 } else { a * 10 })
    }

    pub fn parens(re: Regex, corpus: &Corpus) -> Result<Token> {
        let m_str = re.find(&corpus.text).unwrap().as_str();
        let bytes = m_str.len();
        let lex = match m_str {
            "(" => Lexeme::LParen,
            ")" => Lexeme::RParen,
            _ => unreachable!(),
        };
        Ok(corpus.tag(lex, bytes))
    }

    pub fn true_(re: Regex, corpus: &Corpus) -> Result<Token> {
        let len = re.find(&corpus.text).unwrap().as_str().len();
        Ok(corpus.tag(Lexeme::True, len))
    }

    pub fn false_(re: Regex, corpus: &Corpus) -> Result<Token> {
        let len = re.find(&corpus.text).unwrap().as_str().len();
        Ok(corpus.tag(Lexeme::False, len))
    }

    pub fn string(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(&corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let raw = cs.get(1).unwrap().as_str();
        let text = replace_escapes(raw)?;
        Ok(corpus.tag(Lexeme::String_(text), len))
    }

    pub fn complex_number(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let real_part = if let Some(real_match) = cs.get(1) {
            f64::from_str(real_match.as_str()).unwrap()
        } else {
            0.0
        };
        let imag_part = if let Some(imag_match) = cs.get(2) {
            f64::from_str(imag_match.as_str()).unwrap()
        } else {
            1.0
        };
        let lex = Lexeme::Num(Number::Complex(real_part, imag_part));
        Ok(corpus.tag(lex, len))
    }

    pub fn inf_nan(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let sign = cs.get(1).unwrap().as_str();
        let inf_nan_str = cs.get(2).unwrap().as_str();
        let inf_nan = match (sign, inf_nan_str) {
            (_, "nan") => f64::NAN,
            ("+", "inf") => f64::INFINITY,
            ("-", "inf") => f64::NEG_INFINITY,
            _ => unreachable!(),
        };
        Ok(corpus.tag(Lexeme::Num(Number::Real(inf_nan)), len))
    }

    pub fn real_number(re: Regex, corpus: &Corpus) -> Result<Token> {
        let real_str = re.find(corpus.text).unwrap().as_str();
        let real = f64::from_str(&real_str).expect("Real false positive");
        Ok(corpus.tag(Lexeme::Num(Number::Real(real)), real_str.len()))
    }

    pub fn rational_number(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let numerator_str = cs.get(1).unwrap().as_str();
        let denominator_str = cs.get(2).unwrap().as_str();
        let mut numerator = parse_int(numerator_str);
        let mut denominator = parse_int(denominator_str);
        if denominator < 0 {
            numerator *= -1;
            denominator *= -1;
        }
        let rat = Number::Rational(numerator, denominator as u64);
        Ok(corpus.tag(Lexeme::Num(rat), len))
    }

    pub fn integer(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(corpus.text).unwrap();
        let int_str = cs.get(0).unwrap().as_str();
        let len = int_str.len();
        let int = Number::Integral(parse_int(int_str));
        Ok(corpus.tag(Lexeme::Num(int), len))
    }

    pub fn identifier(re: Regex, corpus: &Corpus) -> Result<Token> {
        let text = re.find(corpus.text).unwrap().as_str();
        let lex = Lexeme::Identifier(text.to_owned());
        Ok(corpus.tag(lex, text.len()))
    }

    pub fn whitespace(re: Regex, corpus: &Corpus) -> Result<Token> {
        let len = re.find(corpus.text).unwrap().as_str().len();
        Err(Error::NoToken(len))
    }

    pub fn line_comment(re: Regex, corpus: &Corpus) -> Result<Token> {
        let len = re.find(corpus.text).unwrap().as_str().len();
        Err(Error::NoToken(len))
    }
}

#[test]
fn lexeme_validation() {
    use Lexeme::*;
    use Number::*;
    let first = |v: Result<Vec<Token>, _>| v.unwrap()[0].elem.clone();

    // Lexically significant symbols
    assert_eq!(first(lex("#t")), True);
    assert_eq!(first(lex("#true")), True);
    assert_eq!(first(lex("#f")), False);
    assert_eq!(first(lex("#false")), False);
    assert_eq!(first(lex("(")), LParen);
    assert_eq!(first(lex(")")), RParen);

    // Strings
    assert_eq!(
        first(lex(r#""hello world""#)),
        String_("hello world".to_owned()),
    );
    assert_eq!(
        first(lex(r#""\"hello\" \"world\"""#)),
        String_("\"hello\" \"world\"".to_owned()),
    );
    assert_eq!(
        first(lex(r#""\\\n\r\t\0\"""#)),
        String_("\\\n\r\t\0\"".to_owned()),
    );
    assert_eq!(
        first(lex("\"asdf\\    \n  asdf\"")),
        String_("asdf asdf".to_owned()),
    );

    // Numbers
    assert_eq!(first(lex("3")), Num(Integral(3)));
    assert_eq!(first(lex("+3e3")), Num(Integral(3000)));
    assert_eq!(first(lex("-3.0")), Num(Real(-3.0)));
    assert_eq!(first(lex("3.0e3")), Num(Real(3000.0)));
    assert_eq!(first(lex("3/5")), Num(Rational(3, 5)));
    assert_eq!(first(lex("-3/-5")), Num(Rational(3, 5)));
    assert_eq!(first(lex("5i")), Num(Complex(0.0, 5.0)));
    assert_eq!(first(lex("8.9e7-2.2e-8i")), Num(Complex(8.9e7, -2.2e-8)));

    // Identifiers
    assert_eq!(first(lex("i")), Identifier("i".to_owned()));
    assert_eq!(first(lex("-")), Identifier("-".to_owned()));
    assert_eq!(first(lex("'")), Identifier("'".to_owned()));
    assert_eq!(first(lex("over_9000")), Identifier("over_9000".to_owned()));
    assert_eq!(
        first(lex("Under_33_times")),
        Identifier("Under_33_times".to_owned())
    );
    assert_eq!(
        first(lex("`~<>.,/'[]{}\\|!@#$%^&*_-+=")),
        Identifier("`~<>.,/'[]{}\\|!@#$%^&*_-+=".to_owned()),
    );
}

#[test]
fn lexeme_sequence_validation() {
    use Lexeme::*;
    use Number::*;
    let untag = |v: Result<Vec<Token>, _>| {
        v.unwrap()
            .iter()
            .map(|t| t.elem.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        untag(lex("(+ 5 19)")),
        vec![
            LParen,
            Identifier("+".to_owned()),
            Num(Integral(5)),
            Num(Integral(19)),
            RParen,
        ],
    );
}

#[test]
fn line_and_column_numbers_are_sane() {
    let input = "line_1_col_1\nline_2_col_2\n  line_3_col_3 line_3_col_16\n\nline_5";
    let expected = vec![(1, 1), (2, 1), (3, 3), (3, 16), (5, 1)];
    let actual = lex(input)
        .unwrap()
        .into_iter()
        .map(|t| (t.line, t.col))
        .collect::<Vec<_>>();
    assert_eq!(expected, actual);
}
