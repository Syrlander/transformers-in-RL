# this whole file is taken from https://github.com/Emil2468/DeepLearningCounting 

def split_args(args):
    # splits the arguments in to help args and other, because --help will be
    # parsed by the current parser, even if it should be parsed by some parser
    # in a different script
    help_args = []
    other_args = []
    for arg in args:
        if arg in ["-h", "--help"]:
            help_args.append("-h")
        else:
            other_args.append(arg)
    return other_args, help_args


def update_conf_with_parsed_args(conf, parsed):
    """
    Updates the conf object in-place with the updated parametes from parsed
    """
    for arg in vars(parsed):
        val = getattr(parsed, arg)
        setattr(conf, arg, val)


def parse_remaining_options(remaining, types={}):
    kwargs = { }
    last_opt_arg = None
    for rem_arg in remaining:
        if rem_arg[:2] == "--":
            opt_arg = rem_arg[2:]
            kwargs[opt_arg] = None
            last_opt_arg = opt_arg
        else:
            if last_opt_arg in types:
                if types[last_opt_arg] == bool:
                    # Attempt to parse bool strings
                    if rem_arg == "False":
                        kwargs[last_opt_arg] = False
                    elif rem_arg == "True":
                        kwargs[last_opt_arg] = True
                    else:
                        kwargs[last_opt_arg] = types[last_opt_arg](rem_arg)
                else:
                    kwargs[last_opt_arg] = types[last_opt_arg](rem_arg)
            else:
                kwargs[last_opt_arg] = rem_arg

    return kwargs