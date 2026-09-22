import json
from collections import defaultdict
from ..paths import object_hash

def load_ontology(path):
    with open(path,encoding="utf-8") as f: data=json.load(f)
    for code,row in data.items():
        if row["id"]!=code or not row["name"] or not isinstance(row.get("description"), str):
            raise ValueError(f"Invalid ontology record: {code}")
    return data

def canonicalize_code(raw,ontology):
    matches=[code for code in ontology if str(int(code))==str(int(raw))]
    if len(matches)!=1: raise ValueError(f"Ambiguous/missing CAMEO code: {raw}")
    return matches[0]

def build_descriptions(relation_map,ontology):
    result={}
    for idx,raw in relation_map.items():
        code=canonicalize_code(raw,ontology); row=ontology[code]
        text=f"Event type: {row['name']}."
        if row["description"].strip(): text+=f" Definition: {row['description']}"
        result[str(idx)]={"local_id":idx,"canonical_code":code,"name":row["name"],
                          "description":text,"description_hash":object_hash(text),
                          "definition_missing":not bool(row["description"].strip())}
    return result
