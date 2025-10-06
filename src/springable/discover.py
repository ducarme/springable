from .simulation import simulate_model
from .readwrite import fileio
from .data import examples_csv_models, gallery_csv_models, article_csv_models
import os
import importlib.resources

def run_simulations_from_article(*figure_nb):
    solver_settings = {'fig1': {},
                       'fig3': {},
                       'fig4': {},
                       'fig5': {'radius': 0.005, 'convergence_value': 1e-8, 'detect_mechanism': False}}

    main_save_dir = fileio.mkdir('results_from_article_simulations')

    if not figure_nb:
        figure_nb = (1, 3, 4, 5)

    
    for fig_nb in figure_nb:
        if fig_nb not in (1, 3, 4, 5):
            print(f"No simulations are related to Fig. {fig_nb}")
            print()
            continue

        print(f'Simulation(s) for Fig. {fig_nb}...')
        save_dir = fileio.mkdir(os.path.join(main_save_dir, f"figure_{fig_nb}"))
        article_data = importlib.resources.files(article_csv_models)    

        model_filepaths = [file for file in article_data.iterdir()
                           if file.name.endswith('.csv') and int(file.name[3]) == fig_nb]
        
        model_filepaths = sorted(model_filepaths, key=lambda fl: fl.name)

        for model_path in model_filepaths:
            simulate_model(model_path,
                           os.path.join(save_dir, os.path.splitext(model_path.name)[0]),
                           solver_settings=solver_settings[f'fig{fig_nb}'],
                           print_model_file=True,
                           print_custom_solver_settings=True)
            print()
        print()

        print(f"You have seen all the simulations from Fig. {fig_nb}.")
        print()

    print('Model CSV files from the article can be viewed at:')
    print('https://github.com/ducarme/springable/tree/main/src/springable/data/article_csv_models')
    print()

    print("To run your own simulation, create your own CSV file and run:")
    print("\timport springable.simulation as ss")
    print("\tss.simulate_model('my_spring_model.csv')")


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
