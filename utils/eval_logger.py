import pickle
import os


class Evaluation_Logger():
    def __init__(self, save_path, file_name="eval_data.pkl"):
        self.eval_dict = {}
        self.save_path = save_path
        self.file_name = file_name

    def add_item(self, key, item):
        self.eval_dict[key] = item

    def get_dict(self):
        return self.eval_dict

    def get_item(self, key):
        return self.eval_dict[key]

    def flush(self):
        if os.path.isdir(self.save_path):
            if os.path.isfile(os.path.join(self.save_path, self.file_name)):
                os.remove(os.path.join(self.save_path, self.file_name))
        else:
            os.makedirs(self.save_path)

        pkl_file = open(os.path.join(self.save_path, self.file_name), "wb")
        pickle.dump(self.eval_dict, pkl_file)
        pkl_file.close()


class Evaluation_Loader():
    def __init__(self, save_path, file_name="eval_data.pkl"):
        self.save_path = save_path
        self.file_name = file_name
        pkl_file = open(os.path.join(self.save_path, self.file_name), "wb")
        self.eval_dict = pickle.load(pkl_file)
        pkl_file.close()

    def get_dict(self):
        return self.eval_dict

    def get_item(self, key):
        return self.eval_dict[key]
