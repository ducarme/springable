from .simulation import simulate_model
from .readwrite import fileio
from .data import examples_csv_models, gallery_csv_models
import os
import importlib.resources




def show_examples(filename=None):
    save_dir = fileio.mkdir('examples_results')
    example_data = importlib.resources.files(examples_csv_models)
    filepaths = [file for file in example_data.iterdir() if file.suffix == '.csv']
    filenames = [os.path.basename(filepath) for filepath in filepaths]

    if filename is not None:
        if filename in filenames:
            filenames = [filename]
            filepaths = [filepath for filepath in filepaths if os.path.basename(filepath).endswith(filename)]
        else:
            print(f'The example "{filename}" is unknown.')
            print('Please choose one among the available examples:')
            print(filenames)
            return

    for model_filename, model_path in zip(filenames, filepaths):
        simulate_model(model_path, os.path.join(save_dir, model_filename.split('.')[0]), print_model_file=True)
        print()

    if len(filenames) > 1:
        print("You have seen all the basic examples.")
        print()

    print('Examples CSV files can be viewed at:')
    print('https://github.com/ducarme/springable/tree/main/src/springable/data/examples_csv_models')
    print()

    print("To run your own simulation, create your own CSV file and run:")
    print("\timport springable.simulation as ss")
    print("\tss.simulate_model('my_spring_model.csv')")


def show_gallery(filename=None):
    save_dir = fileio.mkdir('gallery_results')
    example_data = importlib.resources.files(gallery_csv_models)
    filepaths = [file for file in example_data.iterdir() if file.suffix == '.csv']
    filenames = [os.path.basename(filepath) for filepath in filepaths]

    if filename is not None:
        if filename in filenames:
            filenames = [filename]
            filepaths = [filepath for filepath in filepaths if os.path.basename(filepath).endswith(filename)]
        else:
            print(f'The example "{filename}" is unknown.')
            print('Please choose one among the available gallery items:')
            print(filenames)
            return

    for model_filename, model_path in zip(filenames, filepaths):
        simulate_model(model_path, os.path.join(save_dir, model_filename.split('.')[0]), print_model_file=True)
        print()

    if len(filenames) > 1:
        print("You have visited the entire gallery.")
        print()

    print('Gallery CSV files can be viewed at:')
    print('https://github.com/ducarme/springable/tree/main/src/springable/data/gallery_csv_models')
    print()

    print("Inspired? Create your own CSV file and run:")
    print("\timport springable.simulation as ss")
    print("\tss.simulate_model('my_spring_model.csv')")
