from .simulation import simulate_model
from .readwrite import fileio
import os


def show_examples(filename=None):
    save_dir = fileio.mkdir('examples_results')
    folder = os.path.join('src', 'examples-spring-model-CSV-files')
    filenames = os.listdir(folder)

    if filename is not None:
        if filename in filenames:
            filenames = [filename]
        else:
            print(f'The example "{filename}" is unknown.')
            print('Please choose one among the available examples:')
            print(filenames)
            return

    for i, model_filename in enumerate(filenames):
        model_path = os.path.join(folder, model_filename)
        simulate_model(model_path, os.path.join(save_dir, model_filename.split('.')[0]),print_model_file=True)

        if i != len(filenames) - 1:
            print()

    if len(filenames) > 1:
        print()
        print("You have seen all the basic examples.")

    print()
    print("To run your own simulation, create your own CSV file and run:")
    print("\timport spring.simulation as ss")
    print("\tss.simulate_model('my_spring_model.csv')")


def show_gallery(filename):
    save_dir = fileio.mkdir('gallery_results')
    folder = os.path.join('src', 'gallery-spring-model-CSV-files')
    filenames = os.listdir(folder)

    if filename is not None:
        if filename in filenames:
            filenames = [filename]
        else:
            print(f'The example "{filename}" is unknown.')
            print('Please choose one among the available examples:')
            print(filenames)
            return

    for i, model_filename in enumerate(filenames):
        model_path = os.path.join(folder, model_filename)
        simulate_model(model_path, os.path.join(save_dir, model_filename.split('.')[0]),print_model_file=True)

        if i != len(filenames) - 1:
            print()

    if len(filenames) > 1:
        print()
        print("You have seen the entire gallery.")

    print()
    print("Inspired? Create your own CSV file and run:")
    print("\timport spring.simulation as ss")
    print("\tss.simulate_model('my_spring_model.csv')")