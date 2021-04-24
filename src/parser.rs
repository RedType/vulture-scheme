use crate::{error::ParseError, lexer::*, types::*};
use std::sync::Arc;

type Result<T> = std::result::Result<T, ParseError>;

pub fn parse(tokens: Vec<Token>) -> Result<Obj> {
    // workspace
    let mut ws: Vec<Vec<Obj>> = Vec::new();
    for token in tokens {
        match token.elem {
            Lexeme::LParen => ws.push(Vec::new()),
            Lexeme::RParen => ws.push(listify(ws
                .pop()
                .ok_or(ParseError::MismatchedParenthesis(token))?
            )),
            lex => ws
                .last()
                .ok_or(ParseError::TokenOutsideList(token))?
                .push(lex.into()),
        }
    }
}

impl From<Lexeme> for Obj {
    fn from(lex: Lexeme) -> Self {
        match lex {
            Lexeme::LParen | Lexeme::RParen => panic!("undefined"),
            Lexeme::Identifier(name) => Arc::new(ObjType::Symbol(name)),
            Lexeme::Num(number) => number.into(),
            Lexeme::String_(text) => Arc::new(ObjType::String_(text)),
            Lexeme::True => Arc::new(ObjType::Boolean(true)),
            Lexeme::False => Arc::new(ObjType::Boolean(false)),
        }
    }
}

fn listify(stack: Vec<Obj>) -> Obj {
    let mut head = stack.pop().unwrap_or_else(|| Arc::new(ObjType::Nil));
    while let Some(obj) = head.pop() {
        head = Arc::new(ObjType::Pair(o, head));
    }
    head
}
