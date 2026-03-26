def indented_string(s: str, ident: str = "    "):
    return "\n".join([ident + l for l in s.splitlines()])
