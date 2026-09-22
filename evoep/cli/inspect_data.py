import json
from pathlib import Path
from .common import config_parser,get_config
from .. import ROOT
from ..data.readers import inspect_source,read_id_mapping,load_calendar,iter_quadruples,iter_event_text_links,align_event_links
from ..data.ontology import load_ontology,build_descriptions
from ..paths import atomic_write_json

def main(argv=None):
    parser=config_parser("Inspect source data without modifying it")
    args=parser.parse_args(argv);cfg=get_config(args);src=Path(cfg.data_root)/cfg.dataset
    report={"files":inspect_source(cfg.data_root,cfg.dataset),"mode":"full-structure"}
    report["entities"]=len(read_id_mapping(src/"entity2id.txt"))
    report["relations"]=len(build_descriptions(read_id_mapping(src/"relation2id.txt"),
                                  load_ontology(Path(cfg.data_root)/"CAMEO"/"dict_id2ont.json")))
    report["calendar"]=load_calendar(src/(cfg.dataset+".csv"))
    report["rows"]={}
    for split in ("train","valid","test"):
        report["rows"][split]=sum(1 for _ in align_event_links(iter_quadruples(src/(split+".txt")),
                                                          iter_event_text_links(src/(split+"_w_md5s.txt"))))
    output=ROOT/"artifacts"/cfg.dataset/"inspection.json";atomic_write_json(output,report)
    print(json.dumps({"output":str(output),"rows":report["rows"]}))
if __name__=="__main__":main()
