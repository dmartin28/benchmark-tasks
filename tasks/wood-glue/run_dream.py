#!/usr/bin/env python3
"""Run Dream-1 with the walkthrough's adapter, 20 Hz and 320 x 176 observation cameras.

Reads standard Dream credentials. It never uploads submission media or records a server dataset.
"""
import argparse,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default='auto')
    parser.add_argument('--model',default='dreamlabs/dream-1-beta')
    parser.add_argument('--steps',type=int,default=2400)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--rows-per-request',type=int,default=20)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    from dream.credentials import load
    credential=load()
    status=dict(model=args.model,credentials_available=bool(credential.api_key),gpu_launched=False,inference_requests=0)
    if not credential.api_key:
        args.output.mkdir(parents=True,exist_ok=False)
        status.update(status='blocked',reason='DREAM_API_KEY is not configured in the environment or Dream credentials file.')
        (args.output/'preflight.json').write_text(json.dumps(status,indent=2)+'\n')
        raise SystemExit(status['reason'])
    if args.preflight:
        from inspect_robots_dream.session import DreamSession,Episode
        session=DreamSession()
        try:session.check(Episode(model=args.model,record=False))
        finally:session.close()
        args.output.mkdir(parents=True,exist_ok=False)
        status.update(status='ready',model_available=True)
        (args.output/'preflight.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status))
        return
    import run
    run.run(argparse.Namespace(mode='dream',output=args.output,gpu=args.gpu,steps=args.steps,seed=args.seed,
             width=320,height=176,no_record=False,cpu_workers=8,inspection_camera=False,accelerated_bond=False,
             model=args.model,rows_per_request=args.rows_per_request))


if __name__=='__main__':main()
