import json
from .common import config_parser,get_config
from ..data.prepare import prepare_country

def main(argv=None):
    args=config_parser("Prepare canonical EvoEP artifacts").parse_args(argv)
    manifest=prepare_country(get_config(args))
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
