import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


class DataCenterPowerAgent:
    def __init__(self, start_date: str = "2024-01-01", # Default start date
                 interval_minutes: int = 60, # Default interval
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

        self.start_dt_initial = datetime.strptime(start_date, "%Y-%m-%d")
        self._current_time = self.start_dt_initial
        self.interval_minutes = interval_minutes

        self.num_total_servers = num_total_servers
        self.num_total_storage_arrays = num_total_storage_arrays
        self.num_total_network_devices = num_total_network_devices
        self.num_total_ups_units = num_total_ups_units
        self.num_total_lights = num_total_lights
        self.base_outside_temp_celsius = base_outside_temp_celsius
        self.temp_amplitude_daily = temp_amplitude_daily
        self.temp_amplitude_seasonal = temp_amplitude_seasonal
        self.humidity_base_percent = humidity_base_percent
        self.humidity_amplitude_daily = humidity_amplitude_daily
        self.server_idle_power_factor = server_idle_power_factor
        self.server_peak_power_kw_per_server = server_peak_power_kw_per_server
        self.storage_power_kw_per_unit = storage_power_kw_per_unit
        self.network_power_kw_per_device = network_power_kw_per_device
        self.ups_power_kw_per_unit = ups_power_kw_per_unit
        self.pdu_efficiency_loss_factor = pdu_efficiency_loss_factor
        self.light_power_kw_per_unit = light_power_kw_per_unit
        self.cooling_base_factor = cooling_base_factor
        self.cooling_temp_sensitivity = cooling_temp_sensitivity
        self.cooling_fan_speed_max_percent = cooling_fan_speed_max_percent
        self.cooling_fan_speed_min_percent = cooling_fan_speed_min_percent
        self.target_cooling_temp_celsius = target_cooling_temp_celsius
        self.cooling_temp_response_factor = cooling_temp_response_factor
        self.base_internal_heat_load_kw = base_internal_heat_load_kw
        self.cooling_load_factor_temp = cooling_load_factor_temp
        self.cooling_load_factor_humidity = cooling_load_factor_humidity
        self.workload_cooling_effect_factor = workload_cooling_effect_factor
        self.cooling_system_cop = cooling_system_cop 

    def _calculate_time_factors(self, timestamp):
        time_in_hours = (timestamp - self.start_dt_initial).total_seconds() / 3600.0
        hour_of_day = timestamp.hour
        return time_in_hours, hour_of_day

    def sensor_outside_temp(self, time_in_hours, current_timestamp):
        daily_temp_variation = self.temp_amplitude_daily * np.sin(2 * np.pi * time_in_hours / 24)
        seasonal_phase_shift = (current_timestamp - datetime(current_timestamp.year, 1, 1)).days / 365.25
        seasonal_temp_variation = self.temp_amplitude_seasonal * np.sin(
            2 * np.pi * (time_in_hours / (365.25 * 24) + seasonal_phase_shift + 0.5)
        )
        random_temp_noise = np.random.normal(0, 1.0 + 0.5 * abs(np.sin(2 * np.pi * time_in_hours / 24)))
        sensor_temp_data = self.base_outside_temp_celsius + daily_temp_variation + seasonal_temp_variation + random_temp_noise
        return sensor_temp_data

    def sensor_humidity(self, time_in_hours, outside_temp):
        daily_humidity_variation = self.humidity_amplitude_daily * np.sin(2 * np.pi * (time_in_hours / 24) + np.pi/2)
        temp_effect_on_humidity = -0.5 * (outside_temp - self.base_outside_temp_celsius)
        random_humidity_noise = np.random.normal(0, 3.0)
        sensor_humidity_data = np.clip((self.humidity_base_percent + daily_humidity_variation + temp_effect_on_humidity + random_humidity_noise), 0, 100)
        return sensor_humidity_data

    def server(self, outside_temp):
        active_server_base_ratio = 0.8
        active_server_daily_swing = 0.15
        active_server_noise = np.random.normal(0, 0.02)
        temp_effect_on_servers = -0.01 * (outside_temp - self.base_outside_temp_celsius)
        
        servers_data = (self.num_total_servers * (active_server_base_ratio + active_server_daily_swing * np.sin(time.time() * 2 * np.pi / 86400) + temp_effect_on_servers)
                        + self.num_total_servers * active_server_noise).round().astype(int)
        return np.clip(servers_data, int(self.num_total_servers * 0.7), self.num_total_servers)

    def cpu(self, servers):
        cpu_base_utilization = 40
        cpu_daily_swing = 30
        cpu_noise = np.random.normal(0, 5 * (1 + 0.5 * np.maximum(0, np.sin(time.time() * 2 * np.pi / 86400))))
        server_effect_on_cpu = 0.1 * (servers / self.num_total_servers - 0.8) * cpu_daily_swing
        cpu_data = (cpu_base_utilization + cpu_daily_swing * np.sin(time.time() * 2 * np.pi / 86400) + server_effect_on_cpu + cpu_noise).round().astype(int)
        return np.clip(cpu_data, 0, 100)

    def memory(self, cpu_val):
        memory_base_utilization = 50
        memory_daily_swing = 20
        memory_noise = np.random.normal(0, 4 * (1 + 0.5 * np.maximum(0, np.sin(time.time() * 2 * np.pi / 86400))))
        cpu_effect_on_memory = 0.3 * (cpu_val - 50)
        memory_data = (memory_base_utilization + memory_daily_swing * np.sin(time.time() * 2 * np.pi / 86400) + cpu_effect_on_memory + memory_noise).round().astype(int)
        return np.clip(memory_data, 0, 100)

    def server_power(self, servers, cpu_val, outside_temp): # Removed memory_val as it was not directly used in this function
        combined_utilization_factor = (cpu_val) / 100.0 # Simplified for just CPU affecting power more directly
        temp_power_factor = 1 + 0.005 * (outside_temp - self.base_outside_temp_celsius)
        servers_power_kw_internal = (
            servers * self.server_peak_power_kw_per_server *
            (self.server_idle_power_factor + (1 - self.server_idle_power_factor) * combined_utilization_factor) *
            temp_power_factor
        )
        servers_power_kw_internal = np.maximum(0, servers_power_kw_internal + np.random.normal(0, 0.5))
        return servers_power_kw_internal

    def storage_power(self):
        storage_activity_factor = 0.8 + 0.1 * np.sin(time.time() * 2 * np.pi / 86400) + np.random.normal(0, 0.01)
        storage_power = self.num_total_storage_arrays * self.storage_power_kw_per_unit * storage_activity_factor
        return np.maximum(0, storage_power)

    def network_device(self, servers):
        network_base_ratio = 0.95
        network_workload_swing = 0.03
        server_effect = 0.02 * (servers / self.num_total_servers)
        num_network_devices = int(round(np.clip(
            (self.num_total_network_devices * (network_base_ratio + network_workload_swing * np.sin(time.time() * 2 * np.pi / 86400) + server_effect + np.random.normal(0, 0.01))),
            1, self.num_total_network_devices
        )))
        return num_network_devices

    def ups_units(self, it_power_consumption):
        ups_base_ratio = 0.95
        no_of_ups = self.num_total_ups_units # Assume all UPS units are always active
        return no_of_ups

    def lights(self, hour_of_day):
        light_activity_pattern = 1.0 # Always on, or very high activity
        if 6 <= hour_of_day < 20: # Lights are mostly on during typical working hours and a bit before/after
            light_activity_pattern = 1.0 + np.random.normal(0, 0.1) # High activity during the day
        else:
            light_activity_pattern = 0.4 + np.random.normal(0, 0.05) # Still some lights on, but reduced
        
        light_base_ratio = 1.0 # Base ratio is 1.0 as most are on
        no_of_light = int(round(np.clip(
            (self.num_total_lights * light_base_ratio * light_activity_pattern),
            1, self.num_total_lights
        )))
        return no_of_light

    def cooling_system(self, outside_temp, humidity, hour_of_day):
        """
        Calculates cooling power and temperature for a scenario targeting high PUE (>2.0).
        This involves less efficient cooling strategies and higher fixed overheads,
        with specific behaviors based on hour_of_day.
        """
        # A significant base internal heat load, irrespective of IT load
        current_internal_heat_load_kw = self.base_internal_heat_load_kw * 1.5 + np.random.normal(0, 5.0)

        # Environmental cooling demand is more impactful in an inefficient system
        environmental_cooling_demand = (
            (outside_temp - self.base_outside_temp_celsius) * self.cooling_load_factor_temp * 1.5 +
            (humidity - self.humidity_base_percent) * self.cooling_load_factor_humidity * 1.2
        )
        environmental_cooling_demand = np.maximum(0, environmental_cooling_demand)

        # Cooling power calculation, **independent of IT power consumption here**
        # Instead, it's driven by a high base load and environmental factors, plus hour-of-day
        cooling_power_kw_internal = np.maximum(0,
            (current_internal_heat_load_kw * self.cooling_base_factor * 1.5) + # Much higher base cooling
            (environmental_cooling_demand * 1.2) + # Higher cost for environmental factors
            np.random.normal(0, 2.0) # Larger random fluctuations
        )
        
        # Add a significant additional cooling load during peak hours, representing over-cooling or fixed schedules
        if 9 <= hour_of_day <= 18:  # Business hours, assume higher cooling is maintained regardless
            cooling_power_kw_internal += self.num_total_servers * self.server_peak_power_kw_per_server * 0.3 # Simulate fixed, high cooling during work hours
        else: # Off-peak hours, still significant but slightly reduced
            cooling_power_kw_internal += self.num_total_servers * self.server_peak_power_kw_per_server * 0.15 # Still significant overhead

        # Target internal temperature is kept lower than necessary, or less responsive to outside conditions
        # to ensure over-cooling, which contributes to high power consumption.
        base_target_temp_high_pue = 20.0 # Maintain a consistently low target temperature

        # Adjust target temperature based on hour of day (e.g., trying to run colder during peak hours)
        if 9 <= hour_of_day <= 18:
            base_target_temp_high_pue -= 2.0 # Actively cool more during business hours
        else:
            base_target_temp_high_pue += 1.0 # Slightly relax during off-hours, but still not highly efficient

        temp_adjustment_from_outside_temp = (outside_temp - self.base_outside_temp_celsius) * 0.2
        temp_adjustment_from_humidity = (humidity - self.humidity_base_percent) * 0.1

        cooling_temperature = (
            base_target_temp_high_pue +
            temp_adjustment_from_outside_temp +
            temp_adjustment_from_humidity +
            np.random.normal(0, 1.0)
        )
        
        cooling_temperature = np.clip(cooling_temperature, 16.0, 24.0)

        # Max cooling demand calculated assuming very inefficient operation for the ratio
        max_possible_cooling_demand = (self.base_internal_heat_load_kw * self.cooling_base_factor * 2.5) + \
                                      (50 * self.cooling_load_factor_temp * 2.0) + \
                                      (90 * self.cooling_load_factor_humidity * 1.5) + \
                                      (self.num_total_servers * self.server_peak_power_kw_per_server * 0.5) # Even IT load contributes to max demand

        cooling_system_ratio = cooling_power_kw_internal / max_possible_cooling_demand if max_possible_cooling_demand > 0 else 0
        cooling_system_ratio = np.clip(cooling_system_ratio, 0.5, 1)

        return cooling_power_kw_internal, cooling_system_ratio, cooling_temperature

    def fan(self, cooling_system_ratio):
        base_speed = self.cooling_fan_speed_min_percent * 1.5
        fan_speed_range = self.cooling_fan_speed_max_percent - base_speed
        
        ratio_contribution = cooling_system_ratio * fan_speed_range * 1.2

        calculated_fan_speed = base_speed + ratio_contribution + np.random.normal(0, 5)

        fan_speed_percent = np.clip(calculated_fan_speed,
                                     self.cooling_fan_speed_min_percent * 1.2,
                                     self.cooling_fan_speed_max_percent * 1.05)

        return fan_speed_percent

    def generate_single_data_point(self, timestamp_to_generate=None):
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

        it_power_consumption = servers_power_kw_internal + network_devices_total_power_kw + storage_total_power_kw

        no_of_ups = self.ups_units(it_power_consumption)
        ups_total_power_kw = no_of_ups * self.ups_power_kw_per_unit * 1.1

        pdu_efficiency_loss = it_power_consumption * self.pdu_efficiency_loss_factor * 1.5
        pdu_total_power_kw = pdu_efficiency_loss + np.random.normal(0, 0.5)
        pdu_total_power_kw = np.maximum(0, pdu_total_power_kw)
        
        lights_total_power_kw = no_of_light * self.light_power_kw_per_unit * 1.2

        # Pass hour_of_day to cooling_system, it_power_consumption is no longer a direct argument
        cooling_power_kw_internal, cooling_system_ratio, cooling_temperature = self.cooling_system(
            outside_temp, humidity, hour_of_day
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
        #,
        #    'total facility power (Kw)': total_facility_power_kw,
        #    'PUE': pue
        #}