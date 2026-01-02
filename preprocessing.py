import re

# Common C keywords
C_KEYWORDS = [
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
    'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed',
    'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'
]

class Preprocessor:
    @staticmethod
    def remove_comments(code):
        """Remove single-line and multi-line comments."""
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    @staticmethod
    def normalize_whitespace(code):
        """Normalize spaces and remove unnecessary whitespace."""
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'\s*([+\-*/%=&|<>!])\s*', r'\1', code)
        return code.strip()

    @staticmethod
    def normalize_variables(code):
        """Replace variable identifiers with placeholders."""
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
        id_map, id_counter = {}, 1
        for token in tokens:
            if token not in C_KEYWORDS and token not in id_map:
                id_map[token] = f'ID{id_counter}'
                id_counter += 1
        for orig, repl in id_map.items():
            code = re.sub(r'\b' + re.escape(orig) + r'\b', repl, code)
        return code

    @staticmethod
    def preprocess(code):
        """Full preprocessing pipeline."""
        code = Preprocessor.remove_comments(code)
        code = Preprocessor.normalize_whitespace(code)
        code = Preprocessor.normalize_variables(code)
        return code