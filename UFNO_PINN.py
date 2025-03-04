import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.fft
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import load
import numpy as np
import pandas as pd
import os
import time 
import argparse
from sklearn.metrics import mean_absolute_error
import csv

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

import torch
import torch.nn as nn

class Fourier2DLayer(nn.Module):
    def __init__(self, modes, channels):
        super(Fourier2DLayer, self).__init__()
        self.modes = modes
        self.conv1d = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.weights = None

    def forward(self, x):
        device = x.device  # Determine the device of the input tensor
        batch_size, channels, height, width = x.size()

        # Fourier transformation
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))  

        # Initialize weights 
        if self.weights is None or self.weights.size(0) != batch_size:
            self.weights = torch.randn(
                batch_size, channels, self.modes, self.modes, dtype=torch.cfloat, device=device
            )

        # Fourier modification for x1
        out_ft = torch.zeros_like(x_ft)
        torch.einsum('bixy, ioxy->boxy', x_ft[..., :self.modes, :self.modes], self.weights.permute(1, 0, 2, 3))


        # Inverse Fourier transformation for x1
        x1 = torch.fft.irfftn(out_ft, s=(height, width))

        # Conv1d and ReLU for x2
        x2 = x.view(batch_size, channels, -1)  # Reshape for Conv1d
        x2 = self.conv1d(x2)
        x2 = self.relu(x2)
        x2 = x2.view(batch_size, channels, height, width)  # Reshape back to 2D
        x = x1 + x2
    
        return x

class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()
        # Vector projection
        self.vector_projection = nn.Sequential(
            nn.Linear(2, 256 * 256 * 2),
            nn.ReLU()
        )

        # Encoder layers
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
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

        # Decoder layers
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),  # Skip-Connection
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=2, stride=2),  # Skip-Connection
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Conv2d(8, 8, kernel_size=3, padding=1)

    def forward(self, x):

        # Encoder
        x1 = self.encoder_conv1(x)
        x2 = self.encoder_conv2(x1)
        x3 = self.encoder_conv3(x2)
        # Decoder
        x = self.decoder_conv1(x3)
        x = torch.cat([x, x2], dim=1)  # Skip-Connection

        x = self.decoder_conv2(x)
        x = torch.cat([x, x1], dim=1)  # Skip-Connection

        x = self.decoder_conv3(x)

        # Output layer
        output = self.output_layer(x)

        return output

class UFNO(nn.Module):
    def __init__(self):
        super(UFNO, self).__init__()
        # Vector projection
        self.vector_projection = nn.Sequential(
            nn.Linear(2, 256*256*2),
            nn.ReLU()
        )
        # Lifting Layer (fixed to lift to 8 channels using Conv2d)
        self.lifting = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)

        # Fourier Layers 
        self.fourier1 = Fourier2DLayer(modes=128, channels=8)
        self.fourier2 = Fourier2DLayer(modes=128, channels=8)
        self.fourier3 = Fourier2DLayer(modes=128, channels=8)
        self.fourier4 = Fourier2DLayer(modes=128, channels=8)
        self.fourier5 = Fourier2DLayer(modes=128, channels=8)
        self.fourier6 = Fourier2DLayer(modes=128, channels=8)

        # U-Net Layers
        self.unet1 = UNet2D()
        self.unet2 = UNet2D()
        self.unet3 = UNet2D()

        # Linear Projection Layer
        self.projection1 = nn.Conv2d(8, 1, kernel_size=1)  # Reduce to 1 channel

        # Activation
        self.relu = nn.ReLU()

    def forward(self, vector_data, field):
        # Project vector data
        vector_proj = self.vector_projection(vector_data)  # (batch_size, 256*256*2)
        vector_proj = vector_proj.view(-1, 2, 256, 256)  # (batch_size, 2, 256, 256)

        # Concatenate vector projection and field
        x = torch.cat([field, vector_proj], dim=1)  # (batch_size, 1+2, 256, 256)

        x = self.lifting(x)

        # Fourier layers
        x = self.relu(self.fourier1(x))
        x = self.relu(self.fourier2(x))
        x = self.relu(self.fourier3(x))

        # U-Fourier layers
        x1 = self.relu(self.fourier4(x))
        x2 = self.relu(self.unet1(x))
        x = x1 + x2
        x1 = self.relu(self.fourier5(x))
        x2 = self.relu(self.unet2(x))
        x = x1 + x2
        x = self.relu(self.fourier6(x))
        x2 = self.relu(self.unet3(x))
        x = x1 + x2

        x = self.projection1(x)  # Shape: (batch_size, 1, 256, 256)

        return x



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
    # create model
    model = UFNO().to(device)


    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Prepare datasets and data loaders
    combinations_list = extract_combinations(csv_dir)
    train_combinations, test_combinations = train_test_split(combinations_list, test_size=0.2, random_state=42)
    train_dataset = TempFieldDataset(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, train_combinations)
    test_dataset = TempFieldDataset(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, test_combinations)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    #torch.set_num_threads(4)

    def temperature_field_loss(output, field, target_field):
     
     T0=20 #temperature at ground surface (degC)
     depth_centre=400 # Center of the system (m)
     geothermal_gradient= 0.02# number means temperature increases downwards (K/m).
  
     # Mean Squared Error for temperature
     mse_loss = torch.mean((target_field - output) ** 2)
     
     # Calculate the spatial gradients with U-NET Data
     grad_true_x = target_field[:, :, 1:, :] - target_field[:, :, :-1, :]
     grad_true_y = target_field[:, :, :, 1:] - target_field[:, :, :, :-1]

     # Calculate the spatial gradients with FEM Data
     grad_pred_x = output[:, :, 1:, :] - output[:, :, :-1, :]
     grad_pred_y = output[:, :, :, 1:] - output[:, :, :, :-1]
     
     # Gradient error
     grad_mse_x = torch.mean((grad_true_x - grad_pred_x) ** 2)
     grad_mse_y = torch.mean((grad_true_y - grad_pred_y) ** 2)
     grad_mse_loss = grad_mse_x + grad_mse_y
     
     # Create a 256x256 array with individual steps
     y = torch.linspace(-50, 50, steps=256).unsqueeze(1).expand(256, 256).to(target_field.device)

     # Compute boundary temperature according to the function
     boundary_temp = T0 + ((depth_centre - y) * geothermal_gradient)

     # Scale the boundary condition
     boundary_temp_cpu = boundary_temp.cpu()
     boundary_temp_flat = boundary_temp_cpu.flatten().reshape(-1, 1) 
     boundary_temp_scaled_flat = field_scaler.transform(boundary_temp_flat)
     boundary_temp_scaled = boundary_temp_scaled_flat.reshape(256, 256)
     boundary_temp_scaled = torch.tensor(boundary_temp_scaled, device=boundary_temp.device)

     # Set the scaled boundary pressure
     top_boundary = boundary_temp_scaled[0, :]
     bottom_boundary = boundary_temp_scaled[-1, :]
     right_boundary = boundary_temp_scaled[:, -1]
    
     # Compute the boundaries of the output
     output_top = output[:, :, 0, :]
     output_bottom = output[:, :, -1, :]
     output_right = output[:, :, :, -1]
   
     # MSE for the boundaries
     mse_top = torch.mean((top_boundary - output_top) ** 2)
     mse_bottom = torch.mean((bottom_boundary - output_bottom) ** 2)
     mse_right = torch.mean((right_boundary - output_right) ** 2)
     boundary_loss = mse_top + mse_bottom + mse_right
     

     # Error due to temporal derivatives
     delta_t_target = target_field - field
     delta_t_output = output - field
     temporal_grad_loss = torch.mean((delta_t_target - delta_t_output) ** 2)

     # total loss
     loss =  mse_loss+ grad_mse_loss + temporal_grad_loss + boundary_loss
     
     return (loss)
    


    import csv
    import time
    import torch
    import numpy as np
    import os
    from sklearn.metrics import mean_absolute_error

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
            loss = temperature_field_loss(output, field, target_field)
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
                loss = temperature_field_loss(output, field, target_field)
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
            loss = temperature_field_loss(output, field, target_field)
            test_loss += loss.item() * vector_data.size(0)
            test_mae += mean_absolute_error(target_field.cpu().numpy().flatten(), output.cpu().numpy().flatten()) * vector_data.size(0)
            
    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.8f}, Test MAE: {test_mae:.8f}")

    # Speichern der Testergebnisse in die CSV-Datei
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([None, None, None, None, None, None, test_loss, test_mae])




 