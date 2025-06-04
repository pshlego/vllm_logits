# SPDX-License-Identifier: Apache-2.0

# The CLI entrypoint to vLLM.
import signal
import sys

import vllm_logits.entrypoints.cli.benchmark.main
import vllm_logits.entrypoints.cli.collect_env
import vllm_logits.entrypoints.cli.openai
import vllm_logits.entrypoints.cli.run_batch
import vllm_logits.entrypoints.cli.serve
import vllm_logits.version
from vllm_logits.entrypoints.utils import VLLM_LOGITS_SERVE_PARSER_EPILOG, cli_env_setup
from vllm_logits.utils import FlexibleArgumentParser

CMD_MODULES = [
    vllm_logits.entrypoints.cli.openai,
    vllm_logits.entrypoints.cli.serve,
    vllm_logits.entrypoints.cli.benchmark.main,
    vllm_logits.entrypoints.cli.collect_env,
    vllm_logits.entrypoints.cli.run_batch,
]


def register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def main():
    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_LOGITS_SERVE_PARSER_EPILOG,
    )
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=vllm_logits.version.__version__)
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
