
import pandas as pd

# settings
input_folder = "./data/"
output_folder = "./data/"
output_datastore_fname = output_folder + "populationsim.h5"

if __name__ == "__main__":

    tables = {'hh_sample.csv': 'seed_households',
              'per_sample.csv': 'seed_persons',
              'low_controls.csv': 'low_control_data',
              'mid_controls.csv': 'mid_control_data',
              'meta_controls.csv': 'meta_control_data'}

    for file_name, df_name in tables.items():

        print("copying %s to %s" % (file_name, df_name))

        df = pd.read_csv(input_folder + file_name)

        df.to_hdf(output_datastore_fname, df_name)
