import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import load
import numpy as np
import csv
import pandas as pd
import os
import time
import torch.nn.functional as F 

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the number of threads (adjust as needed)
#torch.set_num_threads(20)


class TempFieldDataset(Dataset):
    """
    Dataset class for predicting temperature fields.
    Loads data samples for training or evaluating a neural network.
    """

    def __init__(self, csv_dir, numpy_dir_temp, numpy_dir_pressure, vector_scaler, heat_out_scaler, field_scaler_temp, field_scalar_pressure, combinations):
        """
        Initializes the dataset.

        Args:
            csv_dir (str): Path to the CSV files.
            numpy_dir_temp (str): Path to the NumPy files for temperature fields.
            numpy_dir_pressure (str): Path to the NumPy files for pressure fields.
            vector_scaler (MinMaxScaler): Scaler for the vector data.
            heat_out_scaler (MinMaxScaler): Scaler for the heat output data.
            field_scaler_temp (MinMaxScaler): Scaler for the temperature fields.
            field_scalar_pressure (MinMaxScaler): Scaler for the pressure fields.
            combinations (list of tuples): List of data point combinations (mass, temp, timestep).
        """
        self.csv_dir = csv_dir
        self.numpy_dir_temp = numpy_dir_temp
        self.numpy_dir_pressure = numpy_dir_pressure
        self.vector_scaler = vector_scaler
        self.heat_out_scaler = heat_out_scaler
        self.field_scaler_temp = field_scaler_temp
        self.field_scalar_pressure = field_scalar_pressure
        self.combinations = combinations

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        # Extract mass, temp, and timestep from the combinations
        mass, temp, timestep = self.combinations[idx]

        # Format mass data
        mass_formatted = "{:.0e}".format(float(mass)).replace("e+0", "E").replace("e+", "E")
        csv_file = f"vector_scalar_fluid_mass_{mass_formatted}_temp_{temp}.csv"

        # Check if the CSV file exists
        if not os.path.exists(os.path.join(self.csv_dir, csv_file)):
            raise FileNotFoundError(f"CSV file {csv_file} does not exist")

        df = pd.read_csv(os.path.join(self.csv_dir, csv_file))

        # Check if the current timestep is present in the CSV file
        if int(timestep) not in df['timestep'].values:
            raise ValueError(f"Timestep {timestep} not found in the CSV file {csv_file}")

        # Extract vector data and output
        vector_data = df.loc[df['timestep'] == int(timestep), ['fluid_mass', 'temp']].values[0]
        scalar_output = df.loc[df['timestep'] == int(timestep), 'produced_T'].values[0]

        # Generate filenames for the NumPy files
        numpy_file_temp = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{timestep}.npy"
        numpy_next_file_temp = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{int(timestep)+1}.npy"
        mass_decimal = f"{float(mass):.1f}"  # Pressure field with a decimal point
        numpy_file_pressure = f"druckfeld_mass{mass_decimal}_temp{temp}_timestep{timestep}.npy"

        # Load the temperature and pressure data
        field_path_temp = os.path.join(self.numpy_dir_temp, numpy_file_temp)
        field_next_path_temp = os.path.join(self.numpy_dir_temp, numpy_next_file_temp)
        field_path_pressure = os.path.join(self.numpy_dir_pressure, numpy_file_pressure)

        if not os.path.exists(field_path_temp) or not os.path.exists(field_next_path_temp):
            raise FileNotFoundError(f"NumPy files {numpy_file_temp} or {numpy_next_file_temp} are missing")

        field_temp = np.load(field_path_temp)
        field_next_temp = np.load(field_next_path_temp)
        field_pressure = np.load(field_path_pressure)

        # Scale the data
        vector_data = self.vector_scaler.transform(vector_data.reshape(1, -1))[0]
        field_temp = field_temp.reshape(-1, 1)
        field_temp = self.field_scaler_temp.transform(field_temp).reshape(256, 256)
        field_temp = field_temp[np.newaxis, :, :]  # Add channel dimension

        field_next_temp = field_next_temp.reshape(-1, 1)
        field_next_temp = self.field_scaler_temp.transform(field_next_temp).reshape(256, 256)
        field_next_temp = field_next_temp[np.newaxis, :, :]  # Add channel dimension

        field_pressure = field_pressure.reshape(-1, 1)
        field_pressure = self.field_scalar_pressure.transform(field_pressure).reshape(256, 256)
        field_pressure = field_pressure[np.newaxis, :, :]  # Add channel dimension

        # Convert to tensors
        vector_data = torch.tensor(vector_data, dtype=torch.float32)
        field_temp = torch.tensor(field_temp, dtype=torch.float32)
        field_next_temp = torch.tensor(field_next_temp, dtype=torch.float32)
        field_pressure = torch.tensor(field_pressure, dtype=torch.float32)

        return (vector_data, field_temp, field_pressure), field_next_temp, (temp, mass_formatted)


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
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
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

    def forward(self, vector_data, field_temp, field_pressure):
        # vector_data: (batch_size, 2)
        # field: (batch_size, 1, 256, 256)

        # Project vector data
        vector_proj = self.vector_projection(vector_data)  # (batch_size, 256*256*2)
        vector_proj = vector_proj.view(-1, 2, 256, 256)  # (batch_size, 2, 256, 256)
    
        # Concatenate vector projection and field
        x = torch.cat([field_temp, field_pressure, vector_proj], dim=1)  # (batch_size, 1+2, 256, 256)

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
    numpy_dir_temp = 'NumpyArrays'
    numpy_dir_pressure = 'NumpyArrays'
    scaler_dir = 'Scalers'
    model_dir = 'AI_Models'

    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)

    # Load scalers
    vector_scaler = load(os.path.join(scaler_dir, 'vector_scaler.joblib'))
    heat_out_scaler = load(os.path.join(scaler_dir, 'heat_out_scaler.joblib'))
    field_scaler_temp = load(os.path.join(scaler_dir, 'field_scaler.joblib'))
    field_scaler_pressure = load(os.path.join(scaler_dir, 'field_scaler_pressure.joblib'))

    # Path where the best model will be saved
    model_checkpoint_path = os.path.join(model_dir, 'test_model_only_tempfield_injection1.pth')

    # Create the model
    model = TempFieldModel().to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Prepare datasets and data loaders
    combinations_list = extract_combinations(csv_dir)
    train_combinations, test_combinations = train_test_split(combinations_list, test_size=0.2, random_state=42)
    train_dataset = TempFieldDataset(csv_dir, numpy_dir_temp, numpy_dir_pressure, vector_scaler, heat_out_scaler, field_scaler_temp, field_scaler_pressure,train_combinations)
    test_dataset = TempFieldDataset(csv_dir, numpy_dir_temp, numpy_dir_pressure, vector_scaler, heat_out_scaler, field_scaler_temp, field_scaler_pressure, test_combinations)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)
    
    
    def temperature_field_loss(output_temp, field_temp, field_pressure, target_field_temp, temp, mass_formatted, epoch):
     
     T0=20 #temperature at ground surface (degC)
     depth_centre=400 # Center of the system (m)
     geothermal_gradient= 0.02# number means temperature increases downwards (K/m).
     lmbda = 3 # thermal conductivity (W/mK)
     p = 800 # specific heat capacity cap (J/kgK) 
     cp = 2650 # density cap (kg/m^3)
  
     # Mean Squared Error for temperature
     mse_loss = torch.mean((target_field_temp - output_temp) ** 2)
     
     # Calculate the spatial gradients with U-NET Data
     grad_true_x = target_field_temp[:, :, 1:, :] - target_field_temp[:, :, :-1, :]
     grad_true_y = target_field_temp[:, :, :, 1:] - target_field_temp[:, :, :, :-1]

     # Calculate the spatial gradients with FEM Data
     grad_pred_x = output_temp[:, :, 1:, :] - output_temp[:, :, :-1, :]
     grad_pred_y = output_temp[:, :, :, 1:] - output_temp[:, :, :, :-1]
     
     # Gradient error
     grad_mse_x = torch.mean((grad_true_x - grad_pred_x) ** 2)
     grad_mse_y = torch.mean((grad_true_y - grad_pred_y) ** 2)
     grad_mse_loss = grad_mse_x + grad_mse_y
     
     # Create a 256x256 array with individual steps
     y = torch.linspace(-50, 50, steps=256).unsqueeze(1).expand(256, 256).to(target_field_temp.device)

     # Compute boundary temperature according to the function
     boundary_temp = T0 + ((depth_centre - y) * geothermal_gradient)

     # Scale the boundary condition
     boundary_temp_cpu = boundary_temp.cpu()
     boundary_temp_flat = boundary_temp_cpu.flatten().reshape(-1, 1) 
     boundary_temp_scaled_flat = field_scaler_temp.transform(boundary_temp_flat)
     boundary_temp_scaled = boundary_temp_scaled_flat.reshape(256, 256)
     boundary_temp_scaled = torch.tensor(boundary_temp_scaled, device=boundary_temp.device)

     # Set the scaled boundary pressure
     top_boundary = boundary_temp_scaled[0, :]
     bottom_boundary = boundary_temp_scaled[-1, :]
     right_boundary = boundary_temp_scaled[:, -1]
    
     # Compute the boundaries of the output
     output_top = output_temp[:, :, 0, :]
     output_bottom = output_temp[:, :, -1, :]
     output_right = output_temp[:, :, :, -1]
     
     
    
     # MSE for the boundaries
     mse_top = torch.mean((top_boundary - output_top) ** 2)
     mse_bottom = torch.mean((bottom_boundary - output_bottom) ** 2)
     mse_right = torch.mean((right_boundary - output_right) ** 2)
     boundary_loss = mse_top + mse_bottom + mse_right
     

     # Error due to temporal derivatives
     delta_t_target = target_field_temp - field_temp
     delta_t_output = output_temp - field_temp 
     temporal_grad_loss = torch.mean((delta_t_target - delta_t_output) ** 2)

     

    
     # rescale field_temp
     field_temp_cpu = field_temp.detach().cpu()
     field_temp_reshaped = field_temp_cpu.view(-1, 256 * 256)
     field_temp_transformed = field_scaler_temp.inverse_transform(field_temp_reshaped)
     field_temp_transformed_reshaped = field_temp_transformed.reshape(-1, 256, 256)
     field_temp_original = field_temp_transformed_reshaped.reshape(field_temp_transformed_reshaped.shape[0], 1, 256, 256)
     field_temp_original = torch.tensor(field_temp_original, device=field_temp.device)
     field_temp_original = field_temp_original + 273.15

     
     
     
     # rescale target_field
     target_field_temp_cpu = target_field_temp.detach().cpu()
     target_field_temp_reshaped = target_field_temp_cpu.view(-1, 256 * 256)
     target_field_temp_transformed = field_scaler_temp.inverse_transform(target_field_temp_reshaped)
     target_field_temp_transformed_reshaped = target_field_temp_transformed.reshape(-1, 256, 256)
     target_field_temp_original = target_field_temp_transformed_reshaped.reshape(target_field_temp_transformed_reshaped.shape[0], 1, 256, 256)
     target_field_temp_original = torch.tensor(target_field_temp_original, device=field_temp.device)
     target_field_temp_original = target_field_temp_original + 273.15

     # Calculate dynamic viscosity
     mu = viscosity(field_temp_original)
     
      # rescale output_temp
     output_temp_cpu = output_temp.detach().cpu()
     output_temp_reshaped = output_temp_cpu.view(-1, 256 * 256)
     output_temp_transformed = field_scaler_temp.inverse_transform(output_temp_reshaped)
     output_temp_transformed_reshaped = output_temp_transformed.reshape(-1, 256, 256)
     output_temp_original = output_temp_transformed_reshaped.reshape(output_temp_transformed_reshaped.shape[0], 1, 256, 256)
     output_temp_original = torch.tensor(output_temp_original, device=output_temp.device)
     output_temp_original = output_temp_original + 273.15
     
     # temporal derivative on non scaled date
     delta_t_output_original = output_temp_original-field_temp_original

     # Gradient on non sclaed data
     grad_x_temp_original = field_temp_original[:, :, 1:, :] - field_temp_original[:, :, :-1, :]
     grad_y_temp_original = field_temp_original[:, :, :, 1:] - field_temp_original[:, :, :, :-1]
 
     # Add boundary values for x-differences to match dimensions
     grad_x_temp_original = F.pad(grad_x_temp_original, (0, 0, 0, 1))  # Padding entlang der Höhe (Dimension 2)

     # Add boundary values for y-differences to match dimensions
     grad_y_temp_original = F.pad(grad_y_temp_original, (0, 1, 0, 0))  # Padding entlang der Breite (Dimension 3)
     
     # energy change of rock
     energy_rock_output = (p * cp)/1000 * delta_t_output_original * 0.75 + 0.25 * density_water(temp+273.15) * specific_heat_water(temp+273.15) * delta_t_output_original
     
     # calculate conduction
     conduction_x = -lmbda * grad_x_temp_original
     conduction_y = -lmbda * grad_y_temp_original 
     
     # Rescale field_pressure
     field_pressure_cpu = field_pressure.detach().cpu()
     field_pressure_reshaped = field_pressure_cpu.view(-1, 256 * 256)  
     field_pressure_transformed = field_scaler_pressure.inverse_transform(field_pressure_reshaped) 
     field_pressure_transformed_reshaped = field_pressure_transformed.reshape(-1, 256, 256)  
     field_pressure_original = field_pressure_transformed_reshaped.reshape(field_temp_transformed_reshaped.shape[0], 1, 256, 256)  
     field_pressure_original = torch.tensor(field_pressure_original, device=field_pressure.device)  
    
     # calculate gradient pressure with FEM data
     grad_x_pressure_original = field_pressure_original[:, :, 1:, :] - field_pressure_original[:, :, :-1, :] 
     grad_y_pressure_original = field_pressure_original[:, :, :, 1:] - field_pressure_original[:, :, :, :-1]

     # Add boundary values for x-differences to match dimensions
     grad_x_pressure_original = F.pad(grad_x_pressure_original, (0, 0, 0, 1))  

     # Add boundary values for y-differences to match dimensions
     grad_y_pressure_original = F.pad(grad_y_pressure_original, (0, 1, 0, 0))  

   
     import matplotlib.pyplot as plt

     # Nehme den ersten Batch und den ersten Kanal
     output_folder = "Druckfeld"
     grad_x_slice = grad_y_pressure_original[0, 0, :, :].cpu().numpy()
        
     # Erstelle die Abbildung
     plt.figure(figsize=(6, 5))
     plt.imshow(grad_x_slice, cmap='viridis', aspect='auto')
     plt.colorbar(label='Gradient')
     plt.xlabel('x')
     plt.ylabel('y')
     plt.show()
     
    

     # Create a 256x256 array with individual steps
     y = torch.linspace(-50, 50, steps=256).unsqueeze(1).expand(256, 256).to(target_field_temp.device)

     # Condition for values between 103 and 153
     hor_perm = torch.full_like(y,1E-16)  # Horizontal permeability value for the cap
     ver_perm = torch.full_like(y, 1E-17)  # Vertical permeability value for the cap
    
     # Apply the condition: when the value of y is between 103 and 153
     mask = (y >= 103) & (y <= 153)

     # Adjust permeability values within the range
     hor_perm[mask] = 1E-11  # Horizontal permeability value for the aquifer
     ver_perm[mask] = 2E-12  # Vertical permeability value for the aquifer

     # Calculate darcy_flow
     darcy_flow_x = -hor_perm/mu * grad_x_pressure_original
     darcy_flow_y = -ver_perm/mu * grad_y_pressure_original
     
    
     # Calculate advection
     advection_x = density_water(field_temp_original) * specific_heat_water(field_temp_original) * darcy_flow_x
     advection_y = density_water(field_temp_original) * specific_heat_water(field_temp_original) * darcy_flow_y
     
     # add advection und conduction
     advkonv_x = conduction_x + advection_x
     advkonv_y = conduction_y + advection_y
     
     advkonv_x = F.pad(advkonv_x, (0, 1, 0, 1))  
     advkonv_y = F.pad(advkonv_y, (0, 1, 0, 1)) 

     # Berechnung der Divergenz
     div_pred = (
      advkonv_x[:, :, 1:, :-1] - advkonv_x[:, :, :-1, :-1] +  
      advkonv_y[:, :, :-1, 1:] - advkonv_y[:, :, :-1, :-1]  
     )

     # Create a mask 
     mask = torch.zeros_like(field_pressure_original)
     mask[:, :, 115:141, 0] = 1  # Set the specified range  to 1

     # Add the fixed value only in the specified areas
     Q = (mass_formatted/91)/26 * mask * (field_temp_original-273.15) * specific_heat_water(field_temp_original)

     # Calculate heat transport
     heat_transport_pred = energy_rock_output + div_pred + Q
     print(torch.mean(heat_transport_pred)**2)
     '''
     import matplotlib.pyplot as plt
    
     if epoch % 20 == 0:
        # Nehme den ersten Batch und den ersten Kanal
        output_folder = "Druckfeld"
        grad_x_slice = heat_transport_pred[0, 0, :, :].cpu().numpy()
        
        # Erstelle die Abbildung
        plt.figure(figsize=(6, 5))
        plt.imshow(grad_x_slice, cmap='viridis', aspect='auto')
        plt.colorbar(label='Gradient')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Speichern des Bildes
        save_path = os.path.join(output_folder, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path)
        plt.close()  # Schließt die aktuelle Abbildung, um Speicher zu sparen
        print(f"Bild für Epoche {epoch} gespeichert: {save_path}") 
     '''
    
     
     
     #Scale Heat transport
     min_heat_transport = torch.min(heat_transport_pred)
     max_heat_transport = torch.max(heat_transport_pred)
     heat_transport_scaled = (heat_transport_pred - min_heat_transport) / (max_heat_transport - min_heat_transport)
     
     # loss of heat transport
     loss_heat = torch.mean((heat_transport_scaled)**2)

     # total loss
     loss =  mse_loss+ grad_mse_loss + temporal_grad_loss+ loss_heat + boundary_loss
     print(f"Grad_mse_loss {grad_mse_loss}")
     print(f"mse_loss {mse_loss}")
     print(f"loss_heat {loss_heat}")
     print(f"temporal_grad_loss {temporal_grad_loss}")
     print(f"boundary_loss {boundary_loss}")
     print(f"loss {loss}")
     
     return (loss)

    def density_water(temperature):
     # Calculate density using the given temperature in Kelvin
     density = (-4.34666972179862e-8 * temperature**4 +
               6.84038298861838e-5 * temperature**3 -
               0.0424392901265117 * temperature**2 +
               11.4162111311416 * temperature -
               101.853102036594)

     return density


    def specific_heat_water(temperature):
     # Calculate specific heat using the given temperature in Kelvin
     specific_heat = (1.06064724162268e-5 * temperature**3 -
                     0.000565979736371933 * temperature**2 -
                     2.74881566547099 * temperature +
                     4760.31624187814)

     return specific_heat

    def viscosity(temperature):
     # Calculate viscosity using the given temperature in Kelvin
        mu =  0.06655 * torch.exp(-0.01338 * temperature)
        return mu
    
    

    # Pfad für die Speicherung der Logs
    csv_file_path = "training_logs2.csv"

    # Erstellen der CSV-Datei und Schreiben der Kopfzeile
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Epoch Time", "Test Loss"])

    # Training loop with early stopping and model checkpointing
    num_epochs = 750
    patience = 750  # For early stopping
    best_val_loss = np.inf
    counter = 0
    early_stop = False

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss = 0
        for (vector_data, field_pressure, field_temp), target_field_temp, (temp, mass_formatted) in train_loader:
            vector_data = vector_data.to(device)
            field_pressure = field_pressure.to(device)
            field_temp = field_temp.to(device)
            target_field_temp = target_field_temp.to(device)

            temp = float(temp[0])
            temp = torch.tensor(temp, dtype=torch.float32).to(device)

            mass_formatted = float(mass_formatted[1])
            mass_formatted = torch.tensor(mass_formatted, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            output_temp = model(vector_data, field_temp, field_pressure)
            loss = temperature_field_loss(output_temp, field_temp, field_pressure, target_field_temp, temp, mass_formatted, epoch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * vector_data.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (vector_data, field_pressure, field_temp), target_field_temp, (temp, mass_formatted) in test_loader:
                vector_data = vector_data.to(device)
                field_pressure = field_pressure.to(device)
                field_temp = field_temp.to(device)
                target_field_temp = target_field_temp.to(device)

                temp = float(temp[0])
                temp = torch.tensor(temp, dtype=torch.float32).to(device)

                mass_formatted = float(mass_formatted[1])
                mass_formatted = torch.tensor(mass_formatted, dtype=torch.float32).to(device)

                output_temp = model(vector_data, field_temp, field_pressure)
                loss = temperature_field_loss(output_temp, field_temp, field_pressure, target_field_temp, temp, mass_formatted, epoch)
                val_loss += loss.item() * vector_data.size(0)

        val_loss /= len(test_loader.dataset)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.8f}, "
            f"Validation Loss: {val_loss:.8f}, Time: {epoch_time:.2f} seconds")

        # Speichern der Ergebnisse in die CSV-Datei
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss, epoch_time, None])

        # Early stopping und Modell-Checkpointing
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

    # Speichern des endgültigen Modells
    model_file = os.path.join(model_dir, 'test_model_only_tempfield_injection_final_version1.pth')
    torch.save(model.state_dict(), model_file)

    # Testen des Modells
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (vector_data, field_pressure, field_temp), target_field_temp, (temp, mass_formatted) in test_loader:
            vector_data = vector_data.to(device)
            field_pressure = field_pressure.to(device)
            field_temp = field_temp.to(device)
            target_field_temp = target_field_temp.to(device)

            temp = float(temp[0])
            temp = torch.tensor(temp, dtype=torch.float32).to(device)

            mass_formatted = float(mass_formatted[1])
            mass_formatted = torch.tensor(mass_formatted, dtype=torch.float32).to(device)

            output_temp = model(vector_data, field_temp, field_pressure)
            loss = temperature_field_loss(output_temp, field_temp, field_pressure, target_field_temp, temp, mass_formatted, epoch)
            test_loss += loss.item() * vector_data.size(0)

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.8f}")

    # Speichern der Testergebnisse in die CSV-Datei
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([None, None, None, None, test_loss])
