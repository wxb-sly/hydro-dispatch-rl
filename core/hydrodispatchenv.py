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
    HEAD_NOMINAL = 100.0              # simplified, net
    EFFICIENCY = 0.85                 # overall
    RHO = 1000.0                      # Water density (kg/m³)
    G = 9.81                          # m/s2

    # simulation parameters

    # week is an episode, a hour is a timestep

    HOURS_PER_EPISODE = 24 * 7
    DT_SECONDS = 3600


    def __init__(self, inflow_m3s: float=25.0): # constant for now
        super().__init__() # inherit all attributes from gym.env
        self.inflow_m3s = inflow_m3s

        self.action_space = spaces.Box(low = np.array([self.TURBINE_Q_MIN], dtype = np.float32), high = np.array([self.TURBINE_Q_MAX], dtype = np.float32), dtype = np.float32)

    # [reservoir level, hour of the day, inflow] ~ normalized to 0 and 1
        self.observation_space = spaces.Box(
            low = np.array([0, 0 , 0], dtype =np.float32),
            high = np.array([1, 1, 1], dtype = np.float32),
            dtype = np.float32,
        )

        # initialized in reset, internal state

        self.volume_m3 = None
        self.current_step= None

    def reset(self, seed=None, options=None):
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

        inflow_norm = np.clip(self.inflow_m3s / self.TURBINE_Q_MAX, 0.0, 1.0)

        return np.array([level_norm, hour_of_day, inflow_norm], dtype= np.float32)


    def _get_info(self) -> dict:
        """
        return debug info
        """
        return {
            "volume_m3": self.volume_m3,
            "step": self.current_step,
            "hour_of_day": self.current_step % 24, # type: ignore

        }


    # manually defining .step(action)
    def step(self, action):
        """
        execute one hourly time step
        Volume(t+1) = Volume(t) + Inflow - Discharge - Spill
        """

        discharge_m3s = np.clip(action[0], self.TURBINE_Q_MIN, self.TURBINE_Q_MAX)

        inflow_vol = self.inflow_m3s * self.DT_SECONDS

        available_volume =  self.volume_m3- self.RESERVOIR_MIN_M3 #type: ignore

        max_discharge_volume =  available_volume

        discharge_volume = min(discharge_m3s * self.DT_SECONDS, max_discharge_volume)
        discharge_volume = max(discharge_volume, 0.0)

        actual_discharge_m3s = discharge_volume / self.DT_SECONDS

        self.volume_m3 +=inflow_vol - discharge_volume #type:ignore

        spill_volume = 0.0
        if self.volume_m3 > self.RESERVOIR_MAX_M3:
            spill_volume = self.volume_m3-self.RESERVOIR_MAX_M3
            self.volume_m3 = self.RESERVOIR_MAX_M3

        self.volume_m3 = max(self.volume_m3, self.RESERVOIR_MIN_M3)

        power_watts = (self.RHO * self.G * actual_discharge_m3s * self.HEAD_NOMINAL * self.EFFICIENCY)

        power_mw = power_watts /1_000_000

        reward = power_mw
        # for now no tarrifs just yet

        spill_m3s = spill_volume /self.DT_SECONDS

        if spill_m3s >0.0:
            reward -= spill_m3s * 0.1




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
        })

        return observation, reward, terminated, truncated, info

