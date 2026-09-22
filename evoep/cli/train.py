from .common import config_parser,get_config
from .. import ROOT
from ..training.trainer import Trainer

def main(argv=None):
    p=config_parser("Train a registered forecasting model with the shared engine")
    p.add_argument("--run-id",required=True);p.add_argument("--resume",action="store_true")
    args=p.parse_args(argv)
    if "/" in args.run_id or "\\" in args.run_id or args.run_id in (".",".."):raise ValueError("Invalid run ID")
    trainer=Trainer(get_config(args),ROOT/"runs"/args.run_id,args.resume)
    try:print(trainer.fit())
    finally:trainer.close()
if __name__=="__main__":main()
