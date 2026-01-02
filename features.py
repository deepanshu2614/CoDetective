import numpy as np
import re
import torch
from transformers import RobertaTokenizer, RobertaModel
from pycparser import c_parser, c_ast

_tokenizer = None
_model = None

def get_codebert():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        _model = RobertaModel.from_pretrained("microsoft/codebert-base")
        _model.eval()
    return _tokenizer, _model

def _strip_comments_and_ws(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"\s+", " ", code).strip()
    return code

def ast_stats(code: str) -> np.ndarray:
    """Extract simple structural AST statistics."""
    try:
        parser = c_parser.CParser()
        if "int main" not in code:
            code = "int main(){" + code + "}"
        ast = parser.parse(code)
    except Exception:
        return np.zeros(8, dtype=np.float32)

    counts = {"FuncDef":0, "If":0, "For":0, "While":0, "Return":0, "Decl":0, "BinaryOp":0, "Assignment":0}
    class Visitor(c_ast.NodeVisitor):
        def generic_visit(self, node):
            name = type(node).__name__
            if name in counts: counts[name]+=1
            super().generic_visit(node)
    Visitor().visit(ast)
    return np.array(list(counts.values()), dtype=np.float32)

def codebert_embedding(code: str) -> np.ndarray:
    tokenizer, model = get_codebert()
    cleaned = _strip_comments_and_ws(code)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return emb.astype(np.float32)

def hybrid_features(code: str) -> np.ndarray:
    cb = codebert_embedding(code)
    astf = ast_stats(code)
    return np.concatenate([cb, astf])