"""Strict raw readers. No header inference and no legacy pickle imports."""
import csv
import re
from pathlib import Path
from itertools import zip_longest
from dataclasses import replace
from datetime import datetime
from .schemas import EventRecord

def inspect_source(data_root,country):
    p=Path(data_root)/country
    names=["stat.txt","entity2id.txt","relation2id.txt","md5_list.json","docs_title_paragraph.json",country+".csv"]
    names += [f"{s}{suffix}" for s in ("train","valid","test") for suffix in (".txt","_w_md5s.txt")]
    result={}
    for name in names:
        f=p/name
        if not f.is_file(): raise FileNotFoundError(f)
        result[name]={"bytes":f.stat().st_size}
    return result

def read_id_mapping(path):
    result={}
    with open(path,encoding="utf-8") as f:
        for line in f:
            name,idx=line.rstrip("\r\n").rsplit(maxsplit=1); idx=int(idx)
            if idx<0 or idx in result: raise ValueError(f"Duplicate/invalid ID: {path}")
            result[idx]=name
    if set(result)!=set(range(len(result))): raise ValueError(f"Non-contiguous IDs: {path}")
    return result

def iter_quadruples(path):
    previous=-1
    with open(path,encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            columns=line.split()
            if len(columns)!=4: raise ValueError(f"{path}:{line_no}: expected four fields")
            h,r,o,t=map(int,columns)
            if min(h,r,o,t)<0 or t<previous: raise ValueError(f"{path}:{line_no}: invalid order/ID")
            previous=t
            yield EventRecord(h,r,o,t,source_split=Path(path).stem)

def parse_md5_field(field):
    ids=tuple(sorted(set(x.strip().lower() for x in field.split(",") if x.strip())))
    if any(not re.fullmatch("[0-9a-f]{32}",x) for x in ids): raise ValueError("Invalid MD5 field")
    return ids

def iter_event_text_links(path):
    with open(path,encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            parts=line.rstrip("\r\n").split("\t",4)
            if len(parts)!=5: raise ValueError(f"{path}:{line_no}: expected five TAB fields")
            yield tuple(map(int,parts[:4])),parse_md5_field(parts[4])

def align_event_links(events,links):
    for n,(event,link) in enumerate(zip_longest(events,links),1):
        if event is None or link is None: raise ValueError(f"Event/text row count mismatch at {n}")
        key=(event.head_id,event.relation_id,event.tail_id,event.day)
        if key!=link[0]: raise ValueError(f"Event/text alignment mismatch at {n}")
        yield replace(event,md5s=link[1])

def deduplicate_records(records):
    rows={}
    for e in records:
        key=(e.head_id,e.relation_id,e.tail_id,e.day)
        if key not in rows: rows[key]=[set(),e.source_split]
        rows[key][0].update(e.md5s)
    for idx,key in enumerate(sorted(rows,key=lambda x:(x[3],x[0],x[1],x[2]))):
        md5s,source=rows[key]
        yield EventRecord(*key,tuple(sorted(md5s)),source,idx)

def load_calendar(csv_path):
    dates={}; count=0
    with open(csv_path,encoding="utf-8") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            day=int(row["timid"]); date=row["date"]
            if day in dates and dates[day]!=date: raise ValueError("Conflicting day/date")
            dates[day]=date; count+=1
    if not dates: raise ValueError("Empty calendar")
    origin=datetime.strptime(dates[min(dates)],"%Y%m%d")
    for day,date in dates.items():
        if (datetime.strptime(date,"%Y%m%d")-origin).days!=day-min(dates):
            raise ValueError("timid is not calendar days")
    if set(dates)!=set(range(min(dates),max(dates)+1)): raise ValueError("Unverified missing calendar dates")
    return {"dates":dates,"rows":count,"min_day":min(dates),"max_day":max(dates)}
