"""Streaming article import and episode-dependent document quarantine."""
import json
import sqlite3
import unicodedata
from itertools import zip_longest
from pathlib import Path
from ..paths import resolve_write_path

def iter_json_array(path,chunk_size=65536):
    """Bounded-memory JSON array reader (stdlib fallback, no ijson required)."""
    decoder=json.JSONDecoder()
    with open(path,encoding="utf-8") as f:
        buf=""; pos=0; eof=False; started=False; need_value=True; after_comma=False
        while True:
            if pos>chunk_size:
                buf=buf[pos:]; pos=0
            while pos>=len(buf) and not eof:
                block=f.read(chunk_size); eof=not block; buf+=block
            while pos<len(buf) and buf[pos].isspace(): pos+=1
            if pos>=len(buf):
                if eof: raise ValueError(f"Truncated JSON array: {path}")
                continue
            if not started:
                if buf[pos]!="[": raise ValueError("Expected JSON array")
                pos+=1; started=True; continue
            if buf[pos]=="]":
                if after_comma: raise ValueError("Trailing comma")
                rest=buf[pos+1:]+f.read()
                if rest.strip(): raise ValueError("Trailing JSON data")
                return
            if not need_value:
                if buf[pos]!=",": raise ValueError("Expected comma")
                pos+=1; need_value=True; after_comma=True; continue
            try:
                item,end=decoder.raw_decode(buf,pos)
            except json.JSONDecodeError:
                if eof: raise ValueError(f"Invalid JSON array: {path}")
                block=f.read(chunk_size); eof=not block; buf+=block; continue
            # Numbers could be incomplete at a chunk boundary. Our inputs are strings/lists.
            yield item
            pos=end; need_value=False; after_comma=False

def normalize_article(title,paragraphs):
    if not isinstance(title,str) or not isinstance(paragraphs,list) or any(not isinstance(p,str) for p in paragraphs):
        raise ValueError("Expected [title, paragraph-string-list]")
    return "\n".join(" ".join(unicodedata.normalize("NFKC",x).split())
                     for x in [title]+paragraphs if x.strip())

def iter_articles(md5_path,docs_path):
    seen=set()
    for md5,doc in zip_longest(iter_json_array(md5_path),iter_json_array(docs_path)):
        if md5 is None or doc is None: raise ValueError("MD5/document length mismatch")
        if md5 in seen: raise ValueError("Duplicate article MD5")
        if not isinstance(doc,list) or len(doc)!=2: raise ValueError("Article schema")
        seen.add(md5)
        yield md5,normalize_article(*doc)

def build_article_store(records,source_dir,output_db,spec,availability_path=None):
    path=resolve_write_path(output_db)
    if path.exists(): raise FileExistsError(path)
    conn=sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE articles(md5 TEXT PRIMARY KEY,text TEXT NOT NULL,available_day INTEGER,proxy_day INTEGER,quarantine INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE links(event_id INTEGER NOT NULL,md5 TEXT NOT NULL,PRIMARY KEY(event_id,md5));
    CREATE TABLE associations(md5 TEXT NOT NULL,relation INTEGER NOT NULL,PRIMARY KEY(md5,relation));
    CREATE TABLE proxy(md5 TEXT PRIMARY KEY,day INTEGER NOT NULL);
    """)
    unavailable=set(spec["val_unseen_ids"]+spec["test_unseen_ids"]+spec["excluded_ids"])
    availability={}
    if availability_path:
        with open(availability_path,encoding="utf-8") as f:
            for line in f:
                row=json.loads(line)
                if not row.get("source"): raise ValueError("Availability needs provenance source")
                availability[row["md5"]]=int(row["available_day"])
    for n,event in enumerate(records):
        conn.executemany("INSERT INTO links VALUES(?,?)",[(event.event_id,m) for m in event.md5s])
        conn.executemany("INSERT OR IGNORE INTO associations VALUES(?,?)",[(m,event.relation_id) for m in event.md5s])
        conn.executemany("INSERT INTO proxy VALUES(?,?) ON CONFLICT(md5) DO UPDATE SET day=max(day,excluded.day)",
                         [(m,event.day) for m in event.md5s])
        if n%20000==0: conn.commit()
    conn.commit()
    blocked={r[0] for r in conn.execute("SELECT DISTINCT md5 FROM associations WHERE relation IN ("+
                                      ",".join("?" for _ in unavailable)+")",tuple(unavailable))}
    proxy=dict(conn.execute("SELECT md5,day FROM proxy"))
    batch=[]; count=0
    src=Path(source_dir)
    for md5,text in iter_articles(src/"md5_list.json",src/"docs_title_paragraph.json"):
        batch.append((md5,text,availability.get(md5),proxy.get(md5),int(md5 in blocked))); count+=1
        if len(batch)>=1000:
            conn.executemany("INSERT INTO articles VALUES(?,?,?,?,?)",batch);conn.commit();batch=[]
    if batch: conn.executemany("INSERT INTO articles VALUES(?,?,?,?,?)",batch)
    conn.executescript("CREATE INDEX association_relation ON associations(relation,md5); CREATE INDEX link_md5 ON links(md5);")
    missing=conn.execute("SELECT COUNT(DISTINCT l.md5) FROM links l LEFT JOIN articles a USING(md5) WHERE a.md5 IS NULL").fetchone()[0]
    if missing:
        conn.close(); raise ValueError(f"{missing} linked documents absent from article array")
    verified=conn.execute("SELECT COUNT(*) FROM articles WHERE available_day IS NOT NULL").fetchone()[0]
    conn.commit();conn.close()
    return {"articles":count,"missing_linked_articles":missing,"quarantined_articles":len(blocked),
            "verified_articles":verified}

class ArticleStore:
    def __init__(self,path,policy,max_articles):
        self.conn=sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro",uri=True)
        self.policy=policy;self.max_articles=max_articles

    def select_allowed_articles(self,event_ids,days,pseudo_ids=()):
        """Only returns allowed document content; association labels never reach the model."""
        out={int(e):[] for e in event_ids}
        if self.policy=="disabled" or not len(event_ids): return out
        datecol="a.available_day" if self.policy=="verified_only" else "a.proxy_day"
        p=tuple(map(int,pseudo_ids)); daymap=dict(zip(map(int,event_ids),map(int,days)))
        for start in range(0,len(event_ids),400):
            ids=list(map(int,event_ids[start:start+400]))
            sql=f"SELECT l.event_id,a.md5,a.text,{datecol} FROM links l JOIN articles a USING(md5) WHERE l.event_id IN ({','.join('?' for _ in ids)}) AND a.quarantine=0 AND {datecol} IS NOT NULL"
            args=ids
            if p:
                sql+=f" AND NOT EXISTS(SELECT 1 FROM associations s WHERE s.md5=a.md5 AND s.relation IN ({','.join('?' for _ in p)}))"
                args=ids+list(p)
            sql+=" ORDER BY l.event_id,a.md5"
            for eid,md5,text,day in self.conn.execute(sql,args):
                if day<=daymap[eid] and len(out[eid])<self.max_articles: out[eid].append((md5,text,day))
        return out

    def close(self): self.conn.close()
