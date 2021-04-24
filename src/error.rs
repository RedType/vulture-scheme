use crate::lexer::Token;

#[derive(Debug)]
pub enum LexError {
    NoToken(usize), // handled by lex() internally
    EscapeAtEndOfString(String),
    InvalidEscapeSequence(String),
    NoMatch(String),
    //ReservedSymbol(String, usize),
}

#[derive(Debug)]
pub enum ParseError {
    MismatchedParenthesis(Token),
    TokenOutsideList(Token),
}
