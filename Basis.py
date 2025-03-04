import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import load
import numpy as np
import pandas as pd
import os
import time 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training with different activations.")
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"Starting training from epoch {args.start_epoch}")


# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the number of threads (adjust as needed)
#torch.set_num_threads(20)

# Define the dataset class
class TempFieldDataset(Dataset):
    """
    Dataset class for temperature field prediction.
    It loads data samples for training or evaluating a neural network model.
    """

    def __init__(self, csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, combinations):
        """
        Initialize the dataset.

        Args:
            csv_dir (str): Directory path containing CSV files.
            numpy_dir (str): Directory path containing numpy files.
            vector_scaler (MinMaxScaler): Scaler for vector data.
            heat_out_scaler (MinMaxScaler): Scaler for heat output data.
            field_scaler (MinMaxScaler): Scaler for field data.
            combinations (list of tuples): List of data point combinations (mass, temp, timestep).
        """
        self.csv_dir = csv_dir
        self.numpy_dir = numpy_dir
        self.vector_scaler = vector_scaler
        self.heat_out_scaler = heat_out_scaler
        self.field_scaler = field_scaler
        self.combinations = combinations

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        mass, temp, timestep = self.combinations[idx]

        # Format mass in scientific notation
        mass_formatted = "{:.0e}".format(float(mass)).replace("e+0", "E").replace("e+", "E")
        csv_file = f"vector_scalar_fluid_mass_{mass_formatted}_temp_{temp}.csv"

        # Check if the CSV file exists
        if not os.path.exists(os.path.join(self.csv_dir, csv_file)):
            raise FileNotFoundError(f"CSV file {csv_file} does not exist")

        df = pd.read_csv(os.path.join(self.csv_dir, csv_file))

        # Check if the current timestep is in the dataframe
        if int(timestep) not in df['timestep'].values:
            raise ValueError(f"Timestep {timestep} not in CSV file {csv_file}")

        vector_data = df.loc[df['timestep'] == int(timestep), ['fluid_mass', 'temp']].values[0]
        scalar_output = df.loc[df['timestep'] == int(timestep), 'produced_T'].values[0]
        numpy_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{timestep}.npy"
        numpy_next_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{int(timestep)+1}.npy"

        field_path = os.path.join(self.numpy_dir, numpy_file)
        field_next_path = os.path.join(self.numpy_dir, numpy_next_file)

        if not os.path.exists(field_path) or not os.path.exists(field_next_path):
            raise FileNotFoundError(f"Numpy files {numpy_file} or {numpy_next_file} do not exist")

        field = np.load(field_path)
        field_next = np.load(field_next_path)

        # Scaling the data
        vector_data = self.vector_scaler.transform(vector_data.reshape(1, -1))[0]
        field = field.reshape(-1, 1)
        field = self.field_scaler.transform(field).reshape(256, 256)
        field = field[np.newaxis, :, :]  # Add channel dimension

        field_next = field_next.reshape(-1, 1)
        field_next = self.field_scaler.transform(field_next).reshape(256, 256)
        field_next = field_next[np.newaxis, :, :]  # Add channel dimension

        # Convert to tensors
        vector_data = torch.tensor(vector_data, dtype=torch.float32)
        field = torch.tensor(field, dtype=torch.float32)
        field_next = torch.tensor(field_next, dtype=torch.float32)

        return (vector_data, field), field_next

# Define the model
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU() #conv1
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels) #con2
        )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        x = x + res
        x = self.ReLU(x)
        return x

class TempFieldModel(nn.Module):
    def __init__(self):
        super(TempFieldModel, self).__init__()
        # Vector projection
        self.vector_projection = nn.Sequential(
            nn.Linear(2, 256*256*2),
            nn.ReLU() #linear
        )

        # Encoder layers
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1+2, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU() 
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU() 
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() 
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU() 
        )
      
    
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(12)]
        )


        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() 
        )
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU() 
        )
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU() 
        )
        self.decoder_conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU() 
        )

        # Output layer
        self.output_layer = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, vector_data, field):
        # vector_data: (batch_size, 2)
        # field: (batch_size, 1, 256, 256)

        # Project vector data
        vector_proj = self.vector_projection(vector_data)  # (batch_size, 256*256*2)
        vector_proj = vector_proj.view(-1, 2, 256, 256)  # (batch_size, 2, 256, 256)
    
        # Concatenate vector projection and field
        x = torch.cat([field, vector_proj], dim=1)  # (batch_size, 1+2, 256, 256)

        # Encoder
        x1 = self.encoder_conv1(x)  # (batch_size, 16, 128, 128)
        x2 = self.encoder_conv2(x1)  # (batch_size, 32, 64, 64)
        x3 = self.encoder_conv3(x2)  # (batch_size, 64, 32, 32)
        x4 = self.encoder_conv4(x3)  # (batch_size, 128, 16, 16)

        # Residual blocks
        x = self.res_blocks(x4)  # (batch_size, 128, 16, 16)

        # Decoder
        x = self.decoder_conv1(x)  # (batch_size, 64, 32, 32)
        x = torch.cat([x, x3], dim=1)  # (batch_size, 64+64=128, 32, 32)

        x = self.decoder_conv2(x)  # (batch_size, 32, 64, 64)
        x = torch.cat([x, x2], dim=1)  # (batch_size, 32+32=64, 64, 64)

        x = self.decoder_conv3(x)  # (batch_size, 16, 128, 128)
        x = torch.cat([x, x1], dim=1)  # (batch_size, 16+16=32, 128, 128)

        x = self.decoder_conv4(x)  # (batch_size, 8, 256, 256)

        # Output layer
        output = self.output_layer(x)  # (batch_size, 1, 256, 256)

        return output

# Function to extract combinations
def extract_combinations(csv_dir):
    """
    Extracts unique combinations of fluid mass, temperature, and timestep from CSV files.
    """
    combinations = set()
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            for _, row in df.iterrows():
                fluid_mass = str(float(row['fluid_mass']))
                temp = str(int(row['temp']))
                timestep = str(int(row['timestep']))
                if int(timestep) < 91:
                    combinations.add((fluid_mass, temp, timestep))
    return list(combinations)

# Main script
if __name__ == '__main__':
    # Define paths
    csv_dir = 'Data_Vector_Scalar'
    numpy_dir = 'NumpyArrays'
    scaler_dir = 'Scalers'
    model_dir = 'AI_Models'

    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)

    # Load scalers
    vector_scaler = load(os.path.join(scaler_dir, 'vector_scaler.joblib'))
    heat_out_scaler = load(os.path.join(scaler_dir, 'heat_out_scaler.joblib'))
    field_scaler = load(os.path.join(scaler_dir, 'field_scaler.joblib'))

    # Path where the best model will be saved
    model_checkpoint_path = os.path.join(model_dir, 'test_model_only_tempfield_injection.pth')

    # Create the model
    model = TempFieldModel().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Prepare datasets and data loaders
    combinations_list = extract_combinations(csv_dir)
    train_combinations, test_combinations = train_test_split(combinations_list, test_size=0.2, random_state=42)
    train_dataset = TempFieldDataset(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, train_combinations)
    test_dataset = TempFieldDataset(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, test_combinations)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
    #torch.set_num_threads(4)

    # Pfad für die Speicherung der Logs
    csv_file_path = "training_logs.csv"

    # Erstellen der CSV-Datei und Schreiben der Kopfzeile
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss (MSE)", "Train MAE", "Validation Loss (MSE)", "Validation MAE", "Epoch Time (s)", "Test Loss (MSE)", "Test MAE"])

    # Training loop with early stopping and model checkpointing
    num_epochs = 750
    patience = 500  # For early stopping
    best_val_loss = np.inf
    counter = 0
    early_stop = False

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss, train_mae = 0, 0
        
        for i, ((vector_data, field), target_field) in enumerate(train_loader, start=1):
            vector_data = vector_data.to(device)
            field = field.to(device)
            target_field = target_field.to(device)
            
            optimizer.zero_grad()
            output = model(vector_data, field)
            loss = criterion(output, target_field)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * vector_data.size(0)
            train_mae += mean_absolute_error(target_field.cpu().numpy().flatten(), output.cpu().detach().numpy().flatten()) * vector_data.size(0)
            
            print(i)

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_mae = 0, 0
        
        with torch.no_grad():
            for i, ((vector_data, field), target_field) in enumerate(test_loader, start=1):
                vector_data = vector_data.to(device)
                field = field.to(device)
                target_field = target_field.to(device)
                
                output = model(vector_data, field)
                loss = criterion(output, target_field)
                val_loss += loss.item() * vector_data.size(0)
                val_mae += mean_absolute_error(target_field.cpu().numpy().flatten(), output.cpu().numpy().flatten()) * vector_data.size(0)
                
                print(i)

        val_loss /= len(test_loader.dataset)
        val_mae /= len(test_loader.dataset)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.8f}, Training MAE: {train_mae:.8f}, "
            f"Validation Loss: {val_loss:.8f}, Validation MAE: {val_mae:.8f}, Time: {epoch_time:.2f} seconds")

        # Speichern der Ergebnisse in die CSV-Datei
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_mae, val_loss, val_mae, epoch_time, None, None])

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_checkpoint_path)
            print("Validation loss decreased, saving model.")
        else:
            counter += 1
            print(f"Validation loss did not improve for {counter} epochs.")
            if counter >= patience:
                print("Early stopping.")
                early_stop = True
                break

    # Save the final model
    model_file = os.path.join(model_dir, 'test_model_only_tempfield_injection_final_version.pth')
    torch.save(model.state_dict(), model_file)

    # Evaluate the model on the test set
    # Load the best model
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    test_loss, test_mae = 0, 0

    with torch.no_grad():
        for (vector_data, field), target_field in test_loader:
            vector_data = vector_data.to(device)
            field = field.to(device)
            target_field = target_field.to(device)
            
            output = model(vector_data, field)
            loss = criterion(output, target_field)
            test_loss += loss.item() * vector_data.size(0)
            test_mae += mean_absolute_error(target_field.cpu().numpy().flatten(), output.cpu().numpy().flatten()) * vector_data.size(0)
            
    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.8f}, Test MAE: {test_mae:.8f}")

    # Speichern der Testergebnisse in die CSV-Datei
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([None, None, None, None, None, None, test_loss, test_mae])
