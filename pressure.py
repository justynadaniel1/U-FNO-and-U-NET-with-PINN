import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import load
import csv
import numpy as np
import pandas as pd
import os
import time
import torch.nn.functional as F 
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

    def __init__(self, csv_dir, numpy_dir_pressure, numpy_dir_temp, vector_scaler, heat_out_scaler, field_scaler_temp, field_scaler_pressure, combinations):
        """
        Initialize the dataset.

        Args:
            csv_dir (str): Directory path containing CSV files.
            numpy_dir_pressure (str): Directory path containing pressure numpy files.
            numpy_dir_temp (str): Directory path containing temperature numpy files.
            vector_scaler (MinMaxScaler): Scaler for vector data.
            heat_out_scaler (MinMaxScaler): Scaler for heat output data.
            field_scaler_temp (MinMaxScaler): Scaler for temperature field data.
            field_scaler_pressure (MinMaxScaler): Scaler for pressure field data.
            combinations (list of tuples): List of data point combinations (mass, temp, timestep).
        """
        self.csv_dir = csv_dir
        self.numpy_dir_pressure = numpy_dir_pressure
        self.numpy_dir_temp = numpy_dir_temp
        self.vector_scaler = vector_scaler
        self.heat_out_scaler = heat_out_scaler
        self.field_scaler_temp = field_scaler_temp
        self.field_scaler_pressure = field_scaler_pressure
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
        
        mass_decimal = f"{float(mass):.1f}" 
        vector_data = df.loc[df['timestep'] == int(timestep), ['fluid_mass', 'temp']].values[0]
        scalar_output = df.loc[df['timestep'] == int(timestep), 'produced_T'].values[0]
        numpy_file_pressure = f"druckfeld_mass{mass_decimal}_temp{temp}_timestep{timestep}.npy"
        numpy_next_file_pressure = f"druckfeld_mass{mass_decimal}_temp{temp}_timestep{int(timestep)+1}.npy"
        numpy_file_temp = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{int(timestep)+1}.npy"

        field_path_pressure = os.path.join(self.numpy_dir_pressure, numpy_file_pressure)
        field_next_path_pressure = os.path.join(self.numpy_dir_pressure, numpy_next_file_pressure)
        field_path_temp = os.path.join(self.numpy_dir_temp, numpy_file_temp)

        # Check if the numpy files exist
        if not os.path.exists(field_path_pressure) or not os.path.exists(field_next_path_pressure):
            raise FileNotFoundError(f"Numpy files {numpy_file_pressure} or {numpy_next_file_pressure} do not exist")

        # Load numpy files
        field_pressure = np.load(field_path_pressure)
        field_next_pressure = np.load(field_next_path_pressure)
        field_temp = np.load(field_path_temp)

        # Scale the data
        vector_data = self.vector_scaler.transform(vector_data.reshape(1, -1))[0]
        field_pressure = field_pressure.reshape(-1, 1)
        field_pressure = self.field_scaler_pressure.transform(field_pressure).reshape(256, 256)
        field_pressure = field_pressure[np.newaxis, :, :]  # Add channel dimension
        
        field_next_pressure = field_next_pressure.reshape(-1, 1)
        field_next_pressure = self.field_scaler_pressure.transform(field_next_pressure).reshape(256, 256)
        field_next_pressure = field_next_pressure[np.newaxis, :, :]  # Add channel dimension
    
        field_temp = field_temp.reshape(-1, 1)
        field_temp = self.field_scaler_temp.transform(field_temp).reshape(256, 256)
        field_temp = field_temp[np.newaxis, :, :]  # Add channel dimension

        # Convert to tensors
        vector_data = torch.tensor(vector_data, dtype=torch.float32)
        field_pressure = torch.tensor(field_pressure, dtype=torch.float32)

        # Check for NaN values in x
        if torch.isnan(field_pressure).any():
            print("x contains NaN values!", field_path_pressure)

        field_next_pressure = torch.tensor(field_next_pressure, dtype=torch.float32) 
        field_temp = torch.tensor(field_temp, dtype=torch.float32)

        return (vector_data, field_pressure, field_temp), field_next_pressure, mass_formatted


# Define the model
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU() 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels) 
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
        
        # Project vector data
        vector_proj = self.vector_projection(vector_data)  
        vector_proj = vector_proj.view(-1, 2, 256, 256)  
    
        # Concatenate vector projection and field
        x = torch.cat([field, vector_proj], dim=1)  

        # Encoder
        x1 = self.encoder_conv1(x)  
        x2 = self.encoder_conv2(x1)  
        x3 = self.encoder_conv3(x2)  
        x4 = self.encoder_conv4(x3) 

        # Residual blocks
        x = self.res_blocks(x4)  

        # Decoder
        x = self.decoder_conv1(x)  
        x = torch.cat([x, x3], dim=1) 
        x = self.decoder_conv2(x)  
        x = torch.cat([x, x2], dim=1)  
        x = self.decoder_conv3(x) 
        x = torch.cat([x, x1], dim=1)  
        x = self.decoder_conv4(x)  

        # Output layer
        output = self.output_layer(x)  

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
    numpy_dir_pressure = 'NumpyArrays'
    numpy_dir_temp = 'NumpyArrays'
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
    model_checkpoint_path = os.path.join(model_dir, 'test_model_only_tempfield_injection.pth')

    # Create the model
    model = TempFieldModel().to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Prepare datasets and data loaders
    combinations_list = extract_combinations(csv_dir)
    train_combinations, test_combinations = train_test_split(combinations_list, test_size=0.2, random_state=42)
    train_dataset = TempFieldDataset(csv_dir, numpy_dir_pressure, numpy_dir_temp, vector_scaler, heat_out_scaler, field_scaler_temp, field_scaler_pressure, train_combinations)
    test_dataset = TempFieldDataset(csv_dir, numpy_dir_pressure, numpy_dir_temp, vector_scaler, heat_out_scaler, field_scaler_temp, field_scaler_pressure, test_combinations)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)
    
    

    def pressure_field_loss(output_pressure, field_pressure, field_temp, target_field_pressure, mass_formatted, epoch):
     depth_centre = 400  # Center of the system in meters
     Sm = 5 * 10 **-5  # Specific storage capacity in 1/Pa
     gravity = -9.81  # Gravitational acceleration in m/s^2
    
     # Rescale the target_field_temp
     field_temp_cpu = field_temp.detach().cpu()
     field_temp_reshaped = field_temp_cpu.view(-1, 256 * 256)
     field_temp_transformed = field_scaler_temp.inverse_transform(field_temp_reshaped)
     field_temp_transformed_reshaped = field_temp_transformed.reshape(-1, 256, 256)
     field_temp_original = field_temp_transformed_reshaped.reshape(field_temp_transformed_reshaped.shape[0], 1, 256, 256)
     field_temp_original = torch.tensor(field_temp_original, device=field_temp.device)
     field_temp_original = field_temp_original + 273.15

     # Calculate dynamic viscosity using the output
     mu = viscosity(field_temp_original)
     
     # Mean Squared Error for pressure
     mse_loss = torch.mean((target_field_pressure - output_pressure) ** 2)

     # Calculate the gradients in the x-direction and the y-directions with U-NET Data
     grad_true_x = target_field_pressure[:, :, 1:, :] - target_field_pressure[:, :, :-1, :]
     grad_true_y = target_field_pressure[:, :, :, 1:] - target_field_pressure[:, :, :, :-1]

      # Calculate the gradients in the x-direction and the y-directions with FEM Data
     grad_pred_x = output_pressure[:, :, 1:, :] - output_pressure[:, :, :-1, :]
     grad_pred_y = output_pressure[:, :, :, 1:] - output_pressure[:, :, :, :-1]

     # Add boundary values for x-differences to match dimensions
     grad_pred_x = F.pad(grad_pred_x, (0, 0, 0, 1))  

     # Add boundary values for y-differences to match dimensions
     grad_pred_y = F.pad(grad_pred_y, (0, 1, 0, 0)) 


     # Gradient error
     grad_mse_x = torch.mean((grad_true_x - grad_pred_x) ** 2)
     grad_mse_y = torch.mean((grad_true_y - grad_pred_y) ** 2)
     grad_mse_loss = grad_mse_x + grad_mse_y
     
     # Error due to temporal derivatives
     delta_t_target = target_field_pressure - field_pressure
     delta_t_output = output_pressure - field_pressure
     temporal_grad_loss = torch.mean((delta_t_target - delta_t_output) ** 2)

     # Rescale output_pressure 
     output_pressure_cpu = output_pressure.detach().cpu()
     output_pressure_reshaped = output_pressure_cpu.view(-1, 256 * 256)
     output_pressure_transformed = field_scaler_pressure.inverse_transform(output_pressure_reshaped)
     output_pressure_transformed_reshaped = output_pressure_transformed.reshape(-1, 256, 256)
     output_pressure_original = output_pressure_transformed_reshaped.reshape(field_temp_transformed_reshaped.shape[0], 1, 256, 256)
     output_pressure_original = torch.tensor(output_pressure_original, device=output_pressure.device)
   
     # Rescale field_pressure
     field_pressure_cpu = field_pressure.detach().cpu()
     field_pressure_reshaped = field_pressure_cpu.view(-1, 256 * 256)
     field_pressure_transformed = field_scaler_pressure.inverse_transform(field_pressure_reshaped)
     field_pressure_transformed_reshaped = field_pressure_transformed.reshape(-1, 256, 256)
     field_pressure_original = field_pressure_transformed_reshaped.reshape(field_temp_transformed_reshaped.shape[0], 1, 256, 256)
     field_pressure_original = torch.tensor(field_pressure_original, device=field_pressure.device)
     
     #temporal derivative on non scaled date
     delta_t_output_original =  output_pressure_original - field_pressure_original
    
     # Gradient on non sclaed data
     grad_x_pressure_original = field_pressure_original[:, :, 1:, :] - field_pressure_original[:, :, :-1, :]
     grad_y_pressure_original = field_pressure_original[:, :, :, 1:] - field_pressure_original[:, :, :, :-1]
     
     # Add boundary values for x-differences to match dimensions
     grad_x_pressure_original = F.pad(grad_x_pressure_original, (0, 0, 0, 1))  

     # Add boundary values for y-differences to match dimensions
     grad_y_pressure_original = F.pad(grad_y_pressure_original, (0, 1, 0, 0))  
     
     # Create a 256x256 array with individual steps
     y = torch.linspace(-50, 50, steps=256).unsqueeze(1).expand(256, 256).to(field_pressure.device)

     # Condition for values between 103 and 153
     hor_perm = torch.full_like(y, 1E-16)  # Horizontal permeability value for the cap
     ver_perm = torch.full_like(y, 2E-17)  # Vertical permeability value for the cap
    
     # Apply the condition: when the value of y is between 103 and 153
     mask = (y >= 103) & (y <= 153)

     # Adjust permeability values within the range
     hor_perm[mask] = 1E-11  # Horizontal permeability value for the aquifer
     ver_perm[mask] = 1E-12  # Vertical permeability value for the aquifer

     # Calculate darcy_flow
     darcy_flow_x = -hor_perm/ mu * grad_x_pressure_original
     darcy_flow_y = -ver_perm/ mu * grad_y_pressure_original

     darcy_flow_x = F.pad(darcy_flow_x, (0, 0, 0, 1))  
     darcy_flow_y = F.pad(darcy_flow_y, (0, 1, 0, 0))  

     # Compute divergence 
     div_q_pred = (
         darcy_flow_x[:, :, 1:, :] - darcy_flow_x[:, :, :-1, :] +
         darcy_flow_y[:, :, :, 1:] - darcy_flow_y[:, :, :, :-1]
     )

     # Create a mask 
     mask = torch.zeros_like(field_pressure_original)
     mask[:, :, 115:141, 0] = 1  # Set the specified range to 1

     # Add the fixed value only in the specified areas
     Q = (mass_formatted/91)/26 * mask
     # Fully implement the Darcy equation
     darcy_pred = Sm * delta_t_output_original + div_q_pred + Q
     print(torch.mean(darcy_pred)**2)
   
     min_darcy_pred= torch.min(darcy_pred)
     max_darcy_pred=torch.max(darcy_pred)

     darcy_scaled = (darcy_pred - min_darcy_pred) / (max_darcy_pred - min_darcy_pred)
    
     # Determine Darcy loss
     darcy_equation_loss = torch.mean((darcy_scaled) ** 2)
     
     # Create a 256x256 array with individual steps
     y = torch.linspace(-50, 50, steps=256).unsqueeze(1).expand(256, 256).to(target_field_pressure.device)

     # Compute boundary pressure according to the function
     boundary_press = (y - depth_centre) * 1000 * gravity + 1E5

     # Scale the boundary condition
     boundary_pres_cpu = boundary_press.cpu()
     boundary_pres_flat = boundary_pres_cpu.flatten().reshape(-1, 1)  
     boundary_pres_scaled_flat = field_scaler_pressure.transform(boundary_pres_flat)
     boundary_pres_scaled = boundary_pres_scaled_flat.reshape(256, 256)
     boundary_pres_scaled = torch.tensor(boundary_pres_scaled, device=boundary_press.device)

     # Set the scaled boundary pressure
     top_boundary = boundary_pres_scaled[0, :]
     bottom_boundary = boundary_pres_scaled[-1, :]
     left_boundary = boundary_pres_scaled[:, 0]
     right_boundary = boundary_pres_scaled[:, -1]
    
    
     # Compute the boundaries of the output
     output_top = output_pressure[:, :, 0, :]
     output_bottom = output_pressure[:, :, -1, :]
     output_left = output_pressure[:, :, :, 0]
     output_right = output_pressure[:, :, :, -1]

     # **MSE-Berechnung**
     mse_top = torch.mean((top_boundary - output_top) ** 2)
     mse_bottom = torch.mean((bottom_boundary - output_bottom) ** 2)
     mse_left = torch.mean((left_boundary - output_left ) ** 2)
     mse_right = torch.mean((right_boundary - output_right) ** 2)

     # **Gesamtverlust**
     boundary_loss = mse_top + mse_bottom + mse_left + mse_right

     # Calculate loss
     total_loss = mse_loss + grad_mse_loss + darcy_equation_loss + boundary_loss + temporal_grad_loss
     print(f"MSE loss {mse_loss}")
     print(f"grad loss {grad_mse_loss}")
     print(f"darcy loss {darcy_equation_loss}")
     print(f"boundary loss {boundary_loss}")
     print(f"temperoal_grad loss {temporal_grad_loss}")
     print(f"total loss {total_loss}")
     return total_loss
    
    def viscosity(temperature):
     mu = 0.06655 * torch.exp(-0.01338 * temperature)  # Verwende torch.exp statt np.exp
     return mu

    def density_water(temperature):
    
     # Calculate density using the given temperature in Kelvin
     density = (-4.34666972179862e-8 * temperature**4 +
               6.84038298861838e-5 * temperature**3 -
               0.0424392901265117 * temperature**2 +
               11.4162111311416 * temperature -
               101.853102036594)

     return density



    # Define model, optimizer, and other configurations
    num_epochs = 750
    patience = 750
    best_val_loss = float('inf')
    counter = 0
    early_stop = False
    csv_file_path = "training_logs1.csv"

    # Create the CSV file and write the header
    with open(csv_file_path, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["Epoch", "Train Loss (MSE)", "Train MAE", "Validation Loss (MSE)", 
                        "Validation MAE", "Epoch Time (s)", "Test Loss (MSE)", "Test MAE"])

    # Training Loop
    for epoch in range(num_epochs):
      epoch_start_time = time.time()
      print(f"Epoch {epoch+1}/{num_epochs}")

      # Training phase
      model.train()
      train_loss, train_mae = 0, 0
      for (vector_data, field_pressure, field_temp), target_field_pressure, mass_formatted in train_loader:
         vector_data = vector_data.to(device)
         field_pressure = field_pressure.to(device)
         field_temp = field_temp.to(device)
         target_field_pressure = target_field_pressure.to(device)
         mass_formatted = torch.tensor(float(mass_formatted[0]), dtype=torch.float32).to(device)

         optimizer.zero_grad()
         output_pressure = model(vector_data, field_pressure, field_temp)
         loss = pressure_field_loss(output_pressure, field_pressure, field_temp, 
                                    target_field_pressure, mass_formatted, epoch)
         loss.backward()
         optimizer.step()

         train_loss += loss.item() * vector_data.size(0)
         train_mae += mean_absolute_error(
               target_field_pressure.cpu().numpy().flatten(), 
               output_pressure.cpu().detach().numpy().flatten()
         ) * vector_data.size(0)

      train_loss /= len(train_loader.dataset)
      train_mae /= len(train_loader.dataset)

      # Validation phase
      model.eval()
      val_loss, val_mae = 0, 0
      with torch.no_grad():
         for (vector_data, field_pressure, field_temp), target_field_pressure, mass_formatted in test_loader:
               vector_data = vector_data.to(device)
               field_pressure = field_pressure.to(device)
               field_temp = field_temp.to(device)
               target_field_pressure = target_field_pressure.to(device)
               mass_formatted = torch.tensor(float(mass_formatted[0]), dtype=torch.float32).to(device)

               output_pressure = model(vector_data, field_pressure, field_temp)
               loss = pressure_field_loss(output_pressure, field_pressure, field_temp, 
                                          target_field_pressure, mass_formatted, epoch)

               val_loss += loss.item() * vector_data.size(0)
               val_mae += mean_absolute_error(
                  target_field_pressure.cpu().numpy().flatten(), 
                  output_pressure.cpu().numpy().flatten()
               ) * vector_data.size(0)

      val_loss /= len(test_loader.dataset)
      val_mae /= len(test_loader.dataset)
      epoch_time = time.time() - epoch_start_time

      print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.8f}, Training MAE: {train_mae:.8f}, "
            f"Validation Loss: {val_loss:.8f}, Validation MAE: {val_mae:.8f}, Time: {epoch_time:.2f} seconds")

      # Save to CSV
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

    # Save final model
    model_file = os.path.join(model_dir, 'test_model_final.pth')
    torch.save(model.state_dict(), model_file)

    # Test phase
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    test_loss, test_mae = 0, 0
    with torch.no_grad():
      for (vector_data, field_pressure, field_temp), target_field_pressure, mass_formatted in test_loader:
         vector_data = vector_data.to(device)
         field_pressure = field_pressure.to(device)
         field_temp = field_temp.to(device)
         target_field_pressure = target_field_pressure.to(device)
         mass_formatted = torch.tensor(float(mass_formatted[0]), dtype=torch.float32).to(device)

         output_pressure = model(vector_data, field_pressure, field_temp)
         loss = pressure_field_loss(output_pressure, field_pressure, field_temp, 
                                    target_field_pressure, mass_formatted, epoch)

         test_loss += loss.item() * vector_data.size(0)
         test_mae += mean_absolute_error(
               target_field_pressure.cpu().numpy().flatten(), 
               output_pressure.cpu().numpy().flatten()
         ) * vector_data.size(0)

    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.8f}, Test MAE: {test_mae:.8f}")

    # Save test results to CSV
    with open(csv_file_path, mode='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([None, None, None, None, None, None, test_loss, test_mae])
