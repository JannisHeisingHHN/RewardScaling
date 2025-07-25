from train import start_training
import toml
import multiprocessing as mp
from tqdm import tqdm

# this line is supposed to make tqdm bar placement more consistent, but I'm sceptical
tqdm.set_lock(mp.RLock())


def clone_dict(d: dict):
    '''Clone a dictionary such that every nested dictionary is newly created'''
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = clone_dict(v)
        else:
            out[k] = v

    return out


def run_experiments(settings_paths: list[str]):
    # load and preprocess all settings files
    settings_ready = []
    for i, sp in enumerate(settings_paths):
        try:
            # I don't quite understand why, but I can't pass the dictionary created by toml.load to a subprocess as it can't access
            # the nested dictionaries within. Instead, I have to create a deep copy of the dictionary and pass that.
            # Using copy.deepcopy doesn't work.
            settings = clone_dict(toml.load(sp))
        except FileNotFoundError:
            print(f"Settings file '{sp}' was not found!")
            return

        # set tqdm arguments
        show_tqdm = settings['setup'].get('show_tqdm', True)
        if show_tqdm:
            run_name = settings['setup'].get('mlflow_run_name', "")
            settings['setup']['show_tqdm'] = {
                'desc': run_name,
                'position': i,
                'leave': True,
            }

        settings_ready.append(settings)

    # start processes
    processes: list[mp.Process] = []
    for sr in settings_ready:
        p = mp.Process(target=start_training, args=(sr,))
        p.start()
        processes.append(p)

    # wait for all processes to terminate
    for p in processes:
        p.join()


if __name__ == "__main__":
    # list of settings that are to be executed
    # settings_paths = ["settings_full_plain.toml", "settings_full_scaled.toml", "settings_single_plain.toml", "settings_single_scaled.toml"]
    settings_paths = ["test_settings.toml", "test_settings2.toml"]
    run_experiments(settings_paths)
