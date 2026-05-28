import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HydroDispatchEnv(gym.Env):
    """
    custom gym env for pror, agent controls turbine discharge to max revenue against time of day electricity tarrifs; physics: conservation of mass in reservoir
    """

    # physics constants
    RESERVOIR_MAX_M3 = 5_000_000
    RESERVOIR_MIN_M3 = 500_000        # Dead storage
    TURBINE_Q_MAX = 50.0
    TURBINE_Q_MIN = 0.0               # can shut down
    HEAD_MIN = 30.0              # net
    EFFICIENCY = 0.85                 # overall
    RHO = 1000.0                      # Water density (kg/m³)
    G = 9.81                          # m/s2

    # simulation parameters

    # week is an episode, a hour is a timestep

    HOURS_PER_EPISODE = 24 * 7
    DT_SECONDS = 3600

    #reward/eco
    ## $/MWh

    TARIFF_SCHEDULE = {
        "off_peak": 50.0,     #(22:00 - 06:00)
        "shoulder": 80.0,    #(06:00 - 10:00, 16:00 - 22:00)
        "peak":     120.0,   #(10:00 - 16:00)
    }


    def __init__(self, inflow_data: np.ndarray=None, inflow_m3s = 25.0):
        super().__init__() # inherit all attributes from gym.env

        if inflow_data is not None:
            self.inflow_data = inflow_data.astype(np.float32)
            self.use_historical=True
        else:
            self.inflow_data = None
            self.use_historical = False
            self.inflow_m3s = inflow_m3s

        self.action_space = spaces.Box(low = np.array([self.TURBINE_Q_MIN], dtype = np.float32), high = np.array([self.TURBINE_Q_MAX], dtype = np.float32), dtype = np.float32)

    # [reservoir level, hour of the day, inflow, current tarif]
        self.observation_space = spaces.Box(
            low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high = np.array([1.0, 1.0, 1.0, 5.0], dtype=np.float32), # Tariff can exceed 1.0 due to volatility
            dtype = np.float32,
        )

        # initialized in reset, internal state

        self.volume_m3 = None
        self.current_step= None

    def reset(self, seed=None, options=None): # api contract, to main structure
        """
        reset env to start new episode
        """
        super().reset(seed=seed) # reset fxn alr defined in gym.env
        # we are inheriting it + adding our own properties

        if self.np_random is not None:
            init_frac = self.np_random.uniform(0.3, 0.7) # we might want randomness in starting level for each episodes
        else:
            init_frac = 0.5

        self.volume_m3 =  init_frac * self.RESERVOIR_MAX_M3

        if self.use_historical:
            max_start = len(self.inflow_data) - self.HOURS_PER_EPISODE #type: ignore
            self.inflow_start_idx = self.np_random.integers(0, max(1, max_start))
        else:
            self.inflow_start_idx=0



        self.current_step = 0

        # defining private so agent doesn't 'reward hack'
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self) -> np.ndarray:
        """
        convert internal state to normalized vector [0,1]
        """
        level_norm = (self.volume_m3 - self.RESERVOIR_MIN_M3) / (self.RESERVOIR_MAX_M3 - self.RESERVOIR_MIN_M3) # type: ignore

        level_norm = np.clip(level_norm, 0, 1)
        hour_of_day = (self.current_step % 24) /23.0 # type: ignore

        inflow_norm = np.clip(self._get_current_inflow() / self.TURBINE_Q_MAX, 0.0, 1.0)

        tariff_norm = self._get_tariff_dynamic(self.current_step % 24) / self.TARIFF_SCHEDULE["peak"] # type:ignore



        return np.array([level_norm, hour_of_day, inflow_norm, tariff_norm], dtype=np.float32)


    def _get_info(self) -> dict:
        """
        return debug info
        """
        return {
            "volume_m3": self.volume_m3,
            "step": self.current_step,
            "hour_of_day": self.current_step % 24, # type: ignore


        }

    def _get_current_inflow(self) -> float:
        """get inflow for current time step
        """
        if self.use_historical:
            idx = self.inflow_start_idx + self.current_step #type: ignore
            idx = min(idx, len(self.inflow_data) -1) #type: ignore
            return float(self.inflow_data[idx]) #type: ignore
        else:
            return self.inflow_m3s

    # manually defining .step(action)
    def step(self, action):
        """
        execute one hourly time step
        Volume(t+1) = Volume(t) + Inflow - Discharge - Spill
        """

        discharge_m3s = np.clip(action[0], self.TURBINE_Q_MIN, self.TURBINE_Q_MAX)

        inflow_vol = self._get_current_inflow() * self.DT_SECONDS # type: ignore

        available_volume =  self.volume_m3- self.RESERVOIR_MIN_M3 #type: ignore

        max_discharge_volume =  available_volume

        discharge_volume = min(discharge_m3s * self.DT_SECONDS, max_discharge_volume)
        discharge_volume = max(discharge_volume, 0.0)

        actual_discharge_m3s = discharge_volume / self.DT_SECONDS

        EVAP_RATE_M_PER_HOUR = 0.005 / 24  # 5mm/day → m/hour
        surface_area_m2 = (self.volume_m3 ** (2/3)) * 10  # Rough scaling #type:ignore
        evap_volume = EVAP_RATE_M_PER_HOUR * surface_area_m2

        self.volume_m3 +=inflow_vol - discharge_volume - evap_volume #type:ignore

        # evap losses are non trivial, including it trains agent for passive losses

        spill_volume = 0.0
        if self.volume_m3 > self.RESERVOIR_MAX_M3:
            spill_volume = self.volume_m3-self.RESERVOIR_MAX_M3
            self.volume_m3 = self.RESERVOIR_MAX_M3

        self.volume_m3 = max(self.volume_m3, self.RESERVOIR_MIN_M3)


        level_fraction = (self.volume_m3 - self.RESERVOIR_MIN_M3) / (self.RESERVOIR_MAX_M3 - self.RESERVOIR_MIN_M3)
        level_fraction = np.clip(level_fraction, 0.0, 1.0)
        effective_head = self.HEAD_MIN + level_fraction * 70.0 #(30 to 70)

        power_watts = (self.RHO * self.G * actual_discharge_m3s * effective_head * self.EFFICIENCY)

        power_mw = power_watts /1_000_000

        hour_of_day = (self.current_step -1) % 24 # type:ignore

        tariff = self._get_tariff_dynamic(hour_of_day)
        revenue = power_mw * tariff * 1 # 1 hour duration

        reward=revenue


        spill_m3s = spill_volume /self.DT_SECONDS

        if spill_m3s > 0.0:
            # Only penalize if the agent was negligent (hoarding water while spilling).
            # If they are already running turbines > 95% capacity, they did all they could.
            agent_requested_q = float(action[0])
            if agent_requested_q < (self.TURBINE_Q_MAX * 0.95):
                # Calculate the volume they COULD have turbined but didn't
                missed_q = min(self.TURBINE_Q_MAX - agent_requested_q, spill_m3s)
                wasted_power = (self.RHO * self.G * missed_q * effective_head * self.EFFICIENCY) / 1_000_000
                reward -= wasted_power * self.TARIFF_SCHEDULE["peak"]

        # also penalize if agent request more than physically possible

        requested_q= float(action[0])
        if requested_q > actual_discharge_m3s + 0.1:
            reward-=10.0


        self.current_step+=1 # type: ignore
        terminated = self.current_step >=self.HOURS_PER_EPISODE
        truncated= False

        observation = self._get_obs()
        info = self._get_info()

        info.update({
            "actual_discharge_m3s": actual_discharge_m3s,
            "requested_discharge_m3s": float(action[0]),
            "power_mw": power_mw,
            "spill_m3s": spill_m3s,
            "evap_volume": evap_volume,
        })

        return observation, float(reward), terminated, truncated, info
    #wrap the final output right at the boundary


    def _get_tariff(self, hour: int) -> float:
        """return the electricity tariff ($/MWh) for a given hour of day"""
        if 22 <= hour or hour < 6:
            return self.TARIFF_SCHEDULE["off_peak"]
        elif 10 <= hour < 16:
            return self.TARIFF_SCHEDULE["peak"]
        else:
            return self.TARIFF_SCHEDULE["shoulder"]


    def _get_tariff_dynamic(self, hour):
        """volatile pricing with random noise."""
        base = self._get_tariff(hour)
        noise = self.np_random.normal(0, base * 0.15)  # 15% volatility
        return max(base + noise, 10.0)

