import os
import shutil


def space_path(*parts):
    """Build and ensure an absolute path to a data subdirectory."""
    path = os.path.join(os.getcwd(), 'data', *parts)
    os.makedirs(path, exist_ok=True)
    return path


def server_to_clients(filename):
    server_space = space_path('server_space')
    client_spaces = [
        space_path('client1_space'),
        space_path('client2_space'),
        space_path('client3_space'),
        space_path('client4_space'),
        space_path('client5_space'),
    ]
    source_file = os.path.join(server_space, filename)
    for client_space in client_spaces:
        shutil.copy(source_file, os.path.join(client_space, filename))


def clients_to_server():
    server_space = space_path('server_space')
    client_specs = [
        ('client1_space', 'client1_weights.h5'),
        ('client2_space', 'client2_weights.h5'),
        ('client3_space', 'client3_weights.h5'),
        ('client4_space', 'client4_weights.h5'),
        ('client5_space', 'client5_weights.h5'),
    ]
    for folder, weight_name in client_specs:
        src = os.path.join(space_path(folder), weight_name)
        dest = os.path.join(server_space, weight_name)
        if os.path.exists(src):
            shutil.move(src, dest)


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def load_final_model_path(client_id):
    """Return the path to a client's trained model file."""
    folder = space_path(f'client{client_id}_space')
    return os.path.join(folder, f'client{client_id}_model.h5')

def get_scaler_path(client_id):
    """Return the path where a client's scaler is saved."""
    folder = space_path(f'client{client_id}_space')
    return os.path.join(folder, f'client{client_id}_scaler.pkl')

def save_inference_results(results, output_path):
    """Save inference result dicts to a CSV file."""
    import csv
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
