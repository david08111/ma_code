
def train_hyperparam():
    from load_config import Config
    from shutil import copyfile
    import os
    import yaml
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="Path to configuration file")
    args = parser.parse_args()

    config = Config()
    config_dict = config(os.path.abspath(args.config))

    if not os.path.isdir(config_dict["settings"]["out_config_path"]):
        os.makedirs(config_dict["settings"]["out_config_path"])


    # def get_node_list(dict, step_list, conf_temp_ls):
    #     if dict:
    #         for key in dict:
    #             # conf_temp_ls.append([key])
    #             if not "settings" in key:
    #                 if "_set" in list(dict[key].keys())[0]:
    #                     step_list.append(key)
    #                     conf_temp_ls.append(step_list.copy())
    #                     step_list.pop()
    #                     break
    #                 else:
    #                     # for key in dict:
    #                     step_list.append(key)
    #                     get_node_list(dict[key], step_list, conf_temp_ls)
    #                 step_list.pop()

    def get_node_list(dict, step_list, conf_temp_ls):
        if dict:
            for key in dict:
                # conf_temp_ls.append([key])
                if not "settings" in key:
                    if "_set" in list(dict[key].keys())[0]:
                        step_list.append(key)
                        conf_temp_ls.append(step_list.copy())
                        step_list.pop()

                    else:
                        # for key in dict:
                        step_list.append(key)
                        get_node_list(dict[key], step_list, conf_temp_ls)
                        step_list.pop()


    def check_item(dict, dependency_list, item):
        t = list(dict[dependency_list[0]].keys())[0]
        if item in list(dict[dependency_list[0]].keys())[0]:
            return dict[dependency_list[0]][list(dict[dependency_list[0]].keys())[0]]
        else:
            # t1 = dict[key]
            # t = list(dict[key].keys())[0]
            # dependency_list.pop(0)
            return check_item(dict[dependency_list[0]], dependency_list[1:], item)

    def get_values(dict, dependency_list):
        if dependency_list:
            return get_values(dict[dependency_list[0]], dependency_list[1:])
        else:
            param_list = list(dict.keys())[1:]
            if "min" in param_list[0] and "max" in param_list[1] and "step" in param_list[2]:
                return np.arange(dict[param_list[0]], dict[param_list[1]] + dict[param_list[2]], dict[param_list[2]])
            else:
                param = []
                for elem in param_list:
                    param.append(dict[elem])
                return param

    def set_value(dict, dependency_list, value):
        if len(dependency_list) > 1:
            set_value(dict[dependency_list[0]], dependency_list[1:], value)
        else:
            dict[dependency_list[0]] = value


    conf_temp_ls = []

    get_node_list(config_dict, [], conf_temp_ls)


    params = []

    params_active = []

    for elem in conf_temp_ls:
        params_active.append(check_item(config_dict, elem, "_set"))

    for i in range(len(params_active) - 1, -1, -1):
        if not params_active[i]:
            # params_active.pop(i)
            conf_temp_ls.pop(i)


    for elem in conf_temp_ls:
        params.append(get_values(config_dict, elem))


    def create_combinations(cfg, out_path, params, config_temp_ls, config_dict):

        # param_list = params.pop()
        # param_path = config_temp_ls.pop()
        param_list = params[-1]
        param_path = config_temp_ls[-1]
        # out_path = out_path + "_" + param_path[-1]
        for param in param_list:
            # if not isinstance(param, str):
            #     param = float(param)
            if  isinstance(param, float):
                param = float(param)
            param_string = str(param).replace(".", "_")
            out_path_temp = out_path + "_" + param_path[-1] + "_" + param_string
            set_value(cfg, param_path, param)
            if len(params[:-1]) > 0:
                create_combinations(cfg, out_path_temp, params[:-1], config_temp_ls[:-1], config_dict)
            else:
                cfg["training"]["save_path"] = os.path.join(config_dict["settings"]["out_save"], os.path.basename(out_path_temp))
                cfg["logging"]["save_path"] = os.path.join(config_dict["settings"]["out_log"], os.path.basename(out_path_temp))
                # cfg["training"]["save_path"] = os.path.join(cfg["training"]["save_path"], os.path.basename(out_path))
                # cfg["logging"]["save_path"] = os.path.join(cfg["logging"]["save_path"], os.path.basename(out_path))
                with open(out_path_temp + ".config", 'w+') as file_handler:
                    yaml.dump(cfg, file_handler, default_flow_style=False, allow_unicode=True,
                              encoding=None)
                file_handler.close()

    for file in [f for f in os.listdir(config_dict["settings"]["in_config_path"]) if os.path.isfile(os.path.join(config_dict["settings"]["in_config_path"], f))]:
        # config_temps_list.append(config(os.path.join(config_dict["settings"]["in_config_path"], file)))
        file_path = os.path.join(config_dict["settings"]["in_config_path"], file)
        cfg_temp = config(file_path)
        fname = os.path.split(file)[-1].rsplit(".", 1)[0]
        out_path = os.path.join(config_dict["settings"]["out_config_path"], fname)
        create_combinations(cfg_temp, out_path, params.copy(), conf_temp_ls.copy(), config_dict)



if __name__ == "__main__":
    train_hyperparam()