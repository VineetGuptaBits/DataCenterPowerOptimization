import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from data_generation_agent import DataCenterPowerAgent

class DataCenterCoolingAgent(DataCenterPowerAgent):
    def __init__(self, start_date: str = "2023-01-01", # Default start date
                 interval_minutes: int = 30, # Default interval
                 num_total_servers: int = 500, # Increased for a larger datacenter
                 num_total_storage_arrays: int = 100, # Increased
                 num_total_network_devices: int = 50, # Increased
                 num_total_ups_units: int = 10, # Increased
                 num_total_lights: int = 100, # Increased
                 base_outside_temp_celsius: float = 25.0,
                 temp_amplitude_daily: float = 5.0,
                 temp_amplitude_seasonal: float = 10.0,
                 humidity_base_percent: float = 60.0,
                 humidity_amplitude_daily: float = 10.0,
                 server_idle_power_factor: float = 0.6,
                 server_peak_power_kw_per_server: float = 0.4, # Slightly reduced IT power to emphasize non-IT
                 storage_power_kw_per_unit: float = 0.15, # Slightly reduced IT power
                 network_power_kw_per_device: float = 0.25, # Slightly reduced IT power
                 ups_power_kw_per_unit: float = 2.5, # **Increased significantly for higher PUE**
                 pdu_efficiency_loss_factor: float = 0.15, # **Increased significantly for higher PUE**
                 light_power_kw_per_unit: float = 0.1, # **Increased for higher PUE**
                 cooling_base_factor: float = 1.5, # **Increased significantly for higher PUE**
                 cooling_temp_sensitivity: float = 0.15,
                 cooling_fan_speed_max_percent: float = 100.0,
                 cooling_fan_speed_min_percent: float = 20.0,
                 target_cooling_temp_celsius: float = 22.0,
                 cooling_temp_response_factor: float = 0.5,
                 base_internal_heat_load_kw: float = 30.0, # **Increased significantly for higher PUE**
                 cooling_load_factor_temp: float = 1.0, # **Increased for higher PUE**
                 cooling_load_factor_humidity: float = 0.4, # **Increased for higher PUE**
                 workload_cooling_effect_factor: float = 5.0,
                 cooling_system_cop: float = 1.5 # Lower COP for less efficient cooling (contributes to high PUE)
                 ):
        super().__init__(start_date, # Default start date
                 interval_minutes, # Default interval
                 num_total_servers, # Increased for a larger datacenter
                 num_total_storage_arrays, # Increased
                 num_total_network_devices, # Increased
                 num_total_ups_units, # Increased
                 num_total_lights, # Increased
                 base_outside_temp_celsius,
                 temp_amplitude_daily,
                 temp_amplitude_seasonal,
                 humidity_base_percent,
                 humidity_amplitude_daily,
                 server_idle_power_factor,
                 server_peak_power_kw_per_server, # Slightly reduced IT power to emphasize non-IT
                 storage_power_kw_per_unit, # Slightly reduced IT power
                 network_power_kw_per_device, # Slightly reduced IT power
                 ups_power_kw_per_unit, # **Increased significantly for higher PUE**
                 pdu_efficiency_loss_factor, # **Increased significantly for higher PUE**
                 light_power_kw_per_unit, # **Increased for higher PUE**
                 cooling_base_factor, # **Increased significantly for higher PUE**
                 cooling_temp_sensitivity,
                 cooling_fan_speed_max_percent,
                 cooling_fan_speed_min_percent,
                 target_cooling_temp_celsius,
                 cooling_temp_response_factor,
                 base_internal_heat_load_kw, # **Increased significantly for higher PUE**
                 cooling_load_factor_temp, # **Increased for higher PUE**
                 cooling_load_factor_humidity, # **Increased for higher PUE**
                 workload_cooling_effect_factor,
                 cooling_system_cop # Lower COP for less efficient cooling (contributes to high PUE)
                 )
        
    def cooling_system_optimized(self, outside_temp, humidity, it_power_consumption):
        """
            Calculates cooling power and temperature, dynamically adjusting based on
            outside conditions and IT power consumption to reduce PUE and power consumption.
        """
            # Calculate environmental heat gain based on outside temperature and humidity
        environmental_heat_gain_kw = (
            (outside_temp - self.base_outside_temp_celsius) * self.cooling_load_factor_temp +
            (humidity - self.humidity_base_percent) * self.cooling_load_factor_humidity
        )
        environmental_heat_gain_kw = np.maximum(0, environmental_heat_gain_kw)

            # Base internal heat load (fixed overhead from non-IT sources like people, building envelope)
        base_internal_heat_load_kw = self.base_internal_heat_load_kw + np.random.normal(0, 0.5)

            # Total heat generated *by IT equipment* that needs to be removed
            # Assume almost all IT power is converted to heat
        it_heat_load_kw = it_power_consumption * 0.99 

            # Total heat load the cooling system needs to handle (in kW of heat)
        total_heat_to_remove_kw = (
                base_internal_heat_load_kw +
                environmental_heat_gain_kw +
                it_heat_load_kw
            )

            # Calculate cooling power required based on total heat to remove and COP
            # Cooling Power (kW) = Heat to Remove (kW) / COP. Higher COP leads to lower PUE.
        cooling_power_kw_internal = total_heat_to_remove_kw / self.cooling_system_cop
        cooling_power_kw_internal = np.maximum(0, cooling_power_kw_internal + np.random.normal(0, 0.5))

            # --- Temperature Calculation Logic (aiming for efficiency and PUE reduction) ---
            # The primary goal is to maintain a safe operating temperature,
            # but also to allow it to float higher when possible to save cooling energy.

        ideal_target_temp = self.target_cooling_temp_celsius

            # Adjust target temperature based on outside conditions
            # If outside is cold, we can potentially run warmer inside, or use free cooling
        temp_adjustment_from_outside_temp = (outside_temp - self.base_outside_temp_celsius) * self.cooling_temp_sensitivity * -0.5 # Negative effect: warmer outside -> slightly lower target needed, but we can also use free cooling
            
            # Adjust target temperature based on IT power consumption:
            # When IT power is low, allow temperature to float higher (save energy).
            # When IT power is high, ensure temperature is kept within acceptable range (but not excessively low).
            
            # Normalized IT power (0 to 1) for subtle temp adjustment
        min_it_power_for_temp_adjust = 0.2 * self.num_total_servers * self.server_peak_power_kw_per_server
        max_it_power_for_temp_adjust = 0.9 * self.num_total_servers * self.server_peak_power_kw_per_server

        normalized_it_power = np.clip(
                (it_power_consumption - min_it_power_for_temp_adjust) /
                (max_it_power_for_temp_adjust - min_it_power_for_temp_adjust),
                0, 1
            )
            
            # This adjustment will be positive (warmer target) for low IT, negative (cooler target) for high IT.
            # But the range should be small, e.g., +/- 2 degrees.
        it_power_temp_adjustment = (1 - normalized_it_power) * 4 - 2 # Range from +2 to -2 degrees

        cooling_temperature = (
                ideal_target_temp +
                temp_adjustment_from_outside_temp + 
                it_power_temp_adjustment +          
                np.random.normal(0, 0.5)
            )
            
            # Enforce a wider but sensible operational range for cooling temperature for PUE optimization
            # Data centers often run hotter than 22C for energy efficiency, e.g., up to 27C.
        cooling_temperature = np.clip(cooling_temperature, 20.0, 27.0) # Adjusted range for better PUE

            # The cooling system ratio should now reflect the *proportion of its max capacity* being used
            # Max theoretical cooling capacity for estimation, based on max heat load scenario
        max_possible_it_heat_load_kw = (self.num_total_servers * self.server_peak_power_kw_per_server * 0.99)
        max_possible_environmental_heat_gain = (
                (self.base_outside_temp_celsius + self.temp_amplitude_seasonal + self.temp_amplitude_daily + 5 - self.base_outside_temp_celsius) * self.cooling_load_factor_temp +
                (self.humidity_base_percent + self.humidity_amplitude_daily + 10 - self.humidity_base_percent) * self.cooling_load_factor_humidity
            )
        max_total_heat_to_remove = self.base_internal_heat_load_kw + max_possible_environmental_heat_gain + max_possible_it_heat_load_kw
            
        max_possible_cooling_power = max_total_heat_to_remove / self.cooling_system_cop if self.cooling_system_cop > 0 else 0

        cooling_system_ratio = cooling_power_kw_internal / max_possible_cooling_power if max_possible_cooling_power > 0 else 0
        cooling_system_ratio = np.clip(cooling_system_ratio, 0, 1)

        return cooling_power_kw_internal, cooling_system_ratio, cooling_temperature
    
    def generate_single_data_point(self, it_power_consumption,timestamp_to_generate=None):
        current_timestamp = timestamp_to_generate if timestamp_to_generate else self._current_time

        time_in_hours, hour_of_day = self._calculate_time_factors(current_timestamp)

        outside_temp = self.sensor_outside_temp(time_in_hours, current_timestamp)
        humidity = self.sensor_humidity(time_in_hours, outside_temp)
        
        servers = self.server(outside_temp)
        cpu_val = self.cpu(servers)
        memory_val = self.memory(cpu_val)
        num_network_devices = self.network_device(servers)
        no_of_light = self.lights(hour_of_day)
        storage_total_power_kw = self.storage_power()

        servers_power_kw_internal = self.server_power(servers, cpu_val, outside_temp)
        network_devices_total_power_kw = num_network_devices * self.network_power_kw_per_device

        #it_power_consumption = servers_power_kw_internal + network_devices_total_power_kw + storage_total_power_kw

        no_of_ups = self.ups_units(it_power_consumption)
        ups_total_power_kw = no_of_ups * self.ups_power_kw_per_unit * 1.1

        pdu_efficiency_loss = it_power_consumption * self.pdu_efficiency_loss_factor * 1.5
        pdu_total_power_kw = pdu_efficiency_loss + np.random.normal(0, 0.5)
        pdu_total_power_kw = np.maximum(0, pdu_total_power_kw)
        
        lights_total_power_kw = no_of_light * self.light_power_kw_per_unit * 1.2

        # Pass hour_of_day to cooling_system, it_power_consumption is no longer a direct argument
        cooling_power_kw_internal, cooling_system_ratio, cooling_temperature = self.cooling_system_optimized(
            outside_temp, humidity, it_power_consumption
        )

        fan_speed = self.fan(
            cooling_system_ratio
        )

        # Calculate Total Facility Power
        total_facility_power_kw = (
            it_power_consumption +
            ups_total_power_kw +
            pdu_total_power_kw +
            lights_total_power_kw +
            cooling_power_kw_internal
        )

        # Calculate PUE
        #pue = total_facility_power_kw / it_power_consumption if it_power_consumption > 0 else 0

        if not timestamp_to_generate:
            self._current_time += timedelta(minutes=self.interval_minutes)

        return {
            'timestamp': current_timestamp,
            'outside temperature': outside_temp,
            'humidity': humidity,
            'servers': servers,
            'cpu': cpu_val,
            'memory': memory_val,
            'num network devices': num_network_devices,
            'no of UPS': no_of_ups,
            'no of light': no_of_light,
            'servers_power_kw_internal': servers_power_kw_internal,
            'storage_total_power_kw': storage_total_power_kw,
            'network devices total power (Kw)': network_devices_total_power_kw,
            'IT power consumption': it_power_consumption,
            'UPS_total_power(Kw)': ups_total_power_kw,
            'PDU_total_power(Kw)': pdu_total_power_kw,
            'lights_total_power(Kw)': lights_total_power_kw,
            'cooling_power_kw_internal': cooling_power_kw_internal,
            'cooling system ratio': cooling_system_ratio,
            'fan speed': fan_speed,
            'cooling temperature': cooling_temperature
        }

"""
#default_start_date = "2025-07-18"
#obj = DataCenterCoolingAgent(start_date=default_start_date,
        interval_minutes=30, # Data every 30 minutes
        num_total_servers=50,
        num_total_storage_arrays=5,
        num_total_network_devices=10,
        num_total_ups_units=2,
        num_total_lights=30,
        base_outside_temp_celsius=25.0,
        temp_amplitude_daily=5.0,
        temp_amplitude_seasonal=8.0,
        humidity_base_percent=60.0,
        humidity_amplitude_daily=10.0,
        server_idle_power_factor=0.5,
        server_peak_power_kw_per_server=0.35,
        storage_power_kw_per_unit=0.4,
        network_power_kw_per_device=0.1,
        ups_power_kw_per_unit=0.5,
        pdu_efficiency_loss_factor=0.02,
        light_power_kw_per_unit=0.05,
        cooling_base_factor=0.3,
        cooling_temp_sensitivity=0.15,
        cooling_fan_speed_max_percent=100.0,
        cooling_fan_speed_min_percent=30.0,
        # NEWLY ADDED PARAMETERS FOR COOLING SYSTEM
        target_cooling_temp_celsius=22.0,
        cooling_temp_response_factor=0.1,
        base_internal_heat_load_kw=10.0, # Example value, adjust as needed
        cooling_load_factor_temp=0.8,    # Example value, adjust as needed
        cooling_load_factor_humidity=0.1, # Example value, adjust as needed
        workload_cooling_effect_factor=5.0 # Example value, adjust as needed
    )

print(obj.generate_single_data_point(it_power_consumption=13, timestamp_to_generate=None))
"""